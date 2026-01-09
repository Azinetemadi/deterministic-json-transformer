#!/usr/bin/env python3
"""
Deterministic JSON Transformer v2.0
===================================
A comprehensive JSON transformation tool with schema validation, filtering,
and deterministic output.

Features:
- Full JSON Schema draft-07 validation (type, enum, const, required, etc.)
- Schema composition (anyOf, oneOf, allOf, not)
- Local $ref resolution
- Numeric constraints (minimum, maximum, multipleOf)
- String constraints (minLength, maxLength, pattern, format)
- Array constraints (minItems, maxItems, uniqueItems)
- JSON Pointer operations with wildcard support
- Include/exclude filtering with rename/move operations
- Default value injection from schema
- Deterministic, canonical JSON output
- Comprehensive error reporting
- Multiple output modes (pretty, compact, canonical)

Usage:
    python transform_v2.py --input data.json --schema schema.json --filter filter.json
    echo '{"id": "1"}' | python transform_v2.py --schema schema.json --pretty
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

# Type aliases
JsonType = Any
JsonObject = Dict[str, Any]
JsonArray = List[Any]
ValidationErrors = List[str]

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Exit Codes
# =============================================================================

class ExitCode(IntEnum):
    """Distinct exit codes for different error types."""
    SUCCESS = 0
    VALIDATION_ERROR = 1
    FILTER_ERROR = 2
    IO_ERROR = 3
    SCHEMA_ERROR = 4
    ARGUMENT_ERROR = 5


# =============================================================================
# Exceptions
# =============================================================================

class TransformError(Exception):
    """Base exception for transformer errors."""
    pass


class PointerError(TransformError):
    """Invalid JSON pointer."""
    pass


class SchemaError(TransformError):
    """Invalid schema."""
    pass


class ValidationError(TransformError):
    """Validation failed."""
    def __init__(self, errors: ValidationErrors):
        self.errors = errors
        super().__init__(f"Validation failed with {len(errors)} error(s)")


class FilterError(TransformError):
    """Filter operation failed."""
    pass


# =============================================================================
# JSON Pointer Operations (RFC 6901)
# =============================================================================

def decode_pointer(pointer: str) -> List[str]:
    """
    Decode a JSON Pointer string into path segments.
    
    Handles RFC 6901 escape sequences:
    - ~0 -> ~
    - ~1 -> /
    
    Args:
        pointer: JSON pointer string (e.g., "/foo/bar/0")
    
    Returns:
        List of path segments
    
    Raises:
        PointerError: If pointer format is invalid
    """
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        raise PointerError(f"JSON pointer must start with '/': {pointer}")
    parts = pointer.split("/")[1:]
    return [part.replace("~1", "/").replace("~0", "~") for part in parts]


def encode_pointer(parts: List[str]) -> str:
    """
    Encode path segments into a JSON Pointer string.
    
    Args:
        parts: List of path segments
    
    Returns:
        JSON pointer string
    """
    if not parts:
        return ""
    escaped = [part.replace("~", "~0").replace("/", "~1") for part in parts]
    return "/" + "/".join(escaped)


def get_by_pointer(doc: JsonType, pointer: str) -> Tuple[bool, JsonType]:
    """
    Retrieve a value from a document using a JSON pointer.
    
    Args:
        doc: The JSON document
        pointer: JSON pointer string
    
    Returns:
        Tuple of (found: bool, value: JsonType)
    """
    if pointer == "":
        return True, doc
    
    try:
        parts = decode_pointer(pointer)
    except PointerError:
        return False, None
    
    current = doc
    for part in parts:
        if isinstance(current, list):
            if not part.isdigit():
                return False, None
            idx = int(part)
            if idx < 0 or idx >= len(current):
                return False, None
            current = current[idx]
        elif isinstance(current, dict):
            if part not in current:
                return False, None
            current = current[part]
        else:
            return False, None
    
    return True, current


def set_by_pointer(doc: JsonType, pointer: str, value: JsonType) -> JsonType:
    """
    Set a value in a document at the specified JSON pointer path.
    Creates intermediate containers as needed.
    
    Args:
        doc: The JSON document (modified in place if possible)
        pointer: JSON pointer string
        value: Value to set
    
    Returns:
        The modified document
    
    Raises:
        PointerError: If the path cannot be created
    """
    if pointer == "":
        return value
    
    parts = decode_pointer(pointer)
    
    # Initialize doc if None
    if doc is None:
        doc = [] if parts[0].isdigit() else {}
    
    current = doc
    for i, part in enumerate(parts):
        is_last = i == len(parts) - 1
        next_part = parts[i + 1] if not is_last else None
        
        if isinstance(current, list):
            if not part.isdigit():
                raise PointerError(f"Expected array index, got '{part}'")
            idx = int(part)
            # Extend array if needed
            while len(current) <= idx:
                current.append(None)
            if is_last:
                current[idx] = value
            else:
                if current[idx] is None:
                    current[idx] = [] if (next_part and next_part.isdigit()) else {}
                current = current[idx]
        
        elif isinstance(current, dict):
            if is_last:
                current[part] = value
            else:
                if part not in current or current[part] is None:
                    current[part] = [] if (next_part and next_part.isdigit()) else {}
                current = current[part]
        
        else:
            raise PointerError(f"Cannot traverse through {type(current).__name__} at '{part}'")
    
    return doc


def delete_by_pointer(doc: JsonType, pointer: str, compact_arrays: bool = False) -> bool:
    """
    Delete a value at the specified JSON pointer path.
    
    Args:
        doc: The JSON document
        pointer: JSON pointer string
        compact_arrays: If True, remove array elements; if False, set to None
    
    Returns:
        True if deletion occurred, False if path not found
    
    Raises:
        PointerError: If attempting to delete root
    """
    if pointer == "":
        raise PointerError("Cannot delete root document")
    
    try:
        parts = decode_pointer(pointer)
    except PointerError:
        return False
    
    current = doc
    for i, part in enumerate(parts[:-1]):
        if isinstance(current, list):
            if not part.isdigit():
                return False
            idx = int(part)
            if idx < 0 or idx >= len(current):
                return False
            current = current[idx]
        elif isinstance(current, dict):
            if part not in current:
                return False
            current = current[part]
        else:
            return False
    
    last_part = parts[-1]
    if isinstance(current, list):
        if not last_part.isdigit():
            return False
        idx = int(last_part)
        if idx < 0 or idx >= len(current):
            return False
        if compact_arrays:
            current.pop(idx)
        else:
            current[idx] = None
        return True
    elif isinstance(current, dict):
        if last_part not in current:
            return False
        del current[last_part]
        return True
    
    return False


def expand_wildcards(doc: JsonType, pointer: str) -> Iterator[str]:
    """
    Expand wildcards in a JSON pointer to concrete paths.
    
    Supports:
    - '*' matches any single key/index
    - '**' matches any depth (recursive)
    
    Args:
        doc: The JSON document
        pointer: JSON pointer with optional wildcards
    
    Yields:
        Concrete JSON pointer strings
    """
    if "*" not in pointer:
        found, _ = get_by_pointer(doc, pointer)
        if found:
            yield pointer
        return
    
    parts = decode_pointer(pointer)
    
    def recurse(current: JsonType, idx: int, path: List[str]) -> Iterator[str]:
        if idx >= len(parts):
            yield encode_pointer(path)
            return
        
        part = parts[idx]
        
        if part == "**":
            # Match zero or more levels
            # First, try matching zero levels (skip **)
            yield from recurse(current, idx + 1, path)
            # Then, try matching one level and continue with **
            if isinstance(current, dict):
                for key in sorted(current.keys()):
                    yield from recurse(current[key], idx, path + [key])
            elif isinstance(current, list):
                for i, item in enumerate(current):
                    yield from recurse(item, idx, path + [str(i)])
        
        elif part == "*":
            # Match exactly one level
            if isinstance(current, dict):
                for key in sorted(current.keys()):
                    yield from recurse(current[key], idx + 1, path + [key])
            elif isinstance(current, list):
                for i, item in enumerate(current):
                    yield from recurse(item, idx + 1, path + [str(i)])
        
        else:
            # Literal match
            if isinstance(current, dict) and part in current:
                yield from recurse(current[part], idx + 1, path + [part])
            elif isinstance(current, list) and part.isdigit():
                i = int(part)
                if 0 <= i < len(current):
                    yield from recurse(current[i], idx + 1, path + [part])
    
    yield from recurse(doc, 0, [])


# =============================================================================
# JSON Schema Validation
# =============================================================================

# Format validators
FORMAT_VALIDATORS: Dict[str, Callable[[str], bool]] = {}


def register_format(name: str) -> Callable:
    """Decorator to register a format validator."""
    def decorator(func: Callable[[str], bool]) -> Callable[[str], bool]:
        FORMAT_VALIDATORS[name] = func
        return func
    return decorator


@register_format("email")
def validate_email(value: str) -> bool:
    """Basic email format validation."""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, value))


@register_format("uri")
def validate_uri(value: str) -> bool:
    """Basic URI format validation."""
    pattern = r"^[a-zA-Z][a-zA-Z0-9+.-]*://[^\s]+$"
    return bool(re.match(pattern, value))


@register_format("date")
def validate_date(value: str) -> bool:
    """ISO 8601 date format (YYYY-MM-DD)."""
    pattern = r"^\d{4}-\d{2}-\d{2}$"
    if not re.match(pattern, value):
        return False
    try:
        year, month, day = map(int, value.split("-"))
        if month < 1 or month > 12:
            return False
        if day < 1 or day > 31:
            return False
        return True
    except ValueError:
        return False


@register_format("date-time")
def validate_datetime(value: str) -> bool:
    """ISO 8601 date-time format."""
    pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?$"
    return bool(re.match(pattern, value))


@register_format("time")
def validate_time(value: str) -> bool:
    """ISO 8601 time format (HH:MM:SS)."""
    pattern = r"^\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?$"
    return bool(re.match(pattern, value))


@register_format("ipv4")
def validate_ipv4(value: str) -> bool:
    """IPv4 address format."""
    parts = value.split(".")
    if len(parts) != 4:
        return False
    for part in parts:
        if not part.isdigit():
            return False
        num = int(part)
        if num < 0 or num > 255:
            return False
    return True


@register_format("ipv6")
def validate_ipv6(value: str) -> bool:
    """Basic IPv6 address format."""
    pattern = r"^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$|^::$|^([0-9a-fA-F]{1,4}:)*:([0-9a-fA-F]{1,4}:)*[0-9a-fA-F]{1,4}$"
    return bool(re.match(pattern, value))


@register_format("uuid")
def validate_uuid(value: str) -> bool:
    """UUID format."""
    pattern = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
    return bool(re.match(pattern, value))


@register_format("hostname")
def validate_hostname(value: str) -> bool:
    """Hostname format."""
    if len(value) > 253:
        return False
    pattern = r"^([a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?\.)*[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?$"
    return bool(re.match(pattern, value))


@dataclass
class ValidationContext:
    """Context for schema validation."""
    root_schema: JsonObject
    errors: ValidationErrors = field(default_factory=list)
    fail_fast: bool = False
    strict_formats: bool = False
    
    def add_error(self, path: str, message: str) -> bool:
        """Add an error and return whether to continue validation."""
        self.errors.append(f"{path or '/'}: {message}")
        return not self.fail_fast
    
    def resolve_ref(self, ref: str) -> Optional[JsonObject]:
        """Resolve a local $ref."""
        if not ref.startswith("#"):
            logger.warning(f"External $ref not supported: {ref}")
            return None
        
        pointer = ref[1:]  # Remove leading #
        found, schema = get_by_pointer(self.root_schema, pointer)
        if not found:
            logger.warning(f"Could not resolve $ref: {ref}")
            return None
        
        return schema


def type_matches(value: JsonType, expected: str) -> bool:
    """Check if a value matches an expected JSON Schema type."""
    type_checks = {
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "string": lambda v: isinstance(v, str),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "null": lambda v: v is None,
    }
    return type_checks.get(expected, lambda _: False)(value)


def validate_schema(
    instance: JsonType,
    schema: JsonObject,
    ctx: ValidationContext,
    path: str = ""
) -> bool:
    """
    Validate an instance against a JSON Schema.
    
    Supports JSON Schema draft-07 keywords.
    
    Args:
        instance: The value to validate
        schema: The JSON Schema
        ctx: Validation context
        path: Current path (for error messages)
    
    Returns:
        True if validation should continue, False to stop
    """
    # Handle boolean schemas
    if isinstance(schema, bool):
        if not schema:
            return ctx.add_error(path, "schema is false (always fails)")
        return True
    
    # Handle $ref
    if "$ref" in schema:
        ref_schema = ctx.resolve_ref(schema["$ref"])
        if ref_schema is None:
            return ctx.add_error(path, f"unresolved $ref: {schema['$ref']}")
        return validate_schema(instance, ref_schema, ctx, path)
    
    # Type validation
    if "type" in schema:
        expected = schema["type"]
        if isinstance(expected, list):
            if not any(type_matches(instance, t) for t in expected):
                if not ctx.add_error(path, f"expected one of {expected}, got {type(instance).__name__}"):
                    return False
        else:
            if not type_matches(instance, expected):
                if not ctx.add_error(path, f"expected {expected}, got {type(instance).__name__}"):
                    return False
    
    # Enum validation
    if "enum" in schema:
        if instance not in schema["enum"]:
            if not ctx.add_error(path, f"value not in enum {schema['enum']}"):
                return False
    
    # Const validation
    if "const" in schema:
        if instance != schema["const"]:
            if not ctx.add_error(path, f"expected const {schema['const']!r}"):
                return False
    
    # Composition keywords
    if "allOf" in schema:
        for i, subschema in enumerate(schema["allOf"]):
            if not validate_schema(instance, subschema, ctx, path):
                return False
    
    if "anyOf" in schema:
        any_valid = False
        original_errors = ctx.errors.copy()
        for subschema in schema["anyOf"]:
            test_ctx = ValidationContext(
                root_schema=ctx.root_schema,
                fail_fast=True,
                strict_formats=ctx.strict_formats
            )
            if validate_schema(instance, subschema, test_ctx, path) and not test_ctx.errors:
                any_valid = True
                break
        if not any_valid:
            ctx.errors = original_errors
            if not ctx.add_error(path, "does not match any schema in anyOf"):
                return False
    
    if "oneOf" in schema:
        match_count = 0
        original_errors = ctx.errors.copy()
        for subschema in schema["oneOf"]:
            test_ctx = ValidationContext(
                root_schema=ctx.root_schema,
                fail_fast=True,
                strict_formats=ctx.strict_formats
            )
            if validate_schema(instance, subschema, test_ctx, path) and not test_ctx.errors:
                match_count += 1
        ctx.errors = original_errors
        if match_count != 1:
            if not ctx.add_error(path, f"must match exactly one schema in oneOf (matched {match_count})"):
                return False
    
    if "not" in schema:
        test_ctx = ValidationContext(
            root_schema=ctx.root_schema,
            fail_fast=True,
            strict_formats=ctx.strict_formats
        )
        if validate_schema(instance, schema["not"], test_ctx, path) and not test_ctx.errors:
            if not ctx.add_error(path, "must not match schema in 'not'"):
                return False
    
    # String-specific validations
    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            if not ctx.add_error(path, f"string length {len(instance)} < minLength {schema['minLength']}"):
                return False
        
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            if not ctx.add_error(path, f"string length {len(instance)} > maxLength {schema['maxLength']}"):
                return False
        
        if "pattern" in schema:
            if not re.search(schema["pattern"], instance):
                if not ctx.add_error(path, f"string does not match pattern '{schema['pattern']}'"):
                    return False
        
        if "format" in schema:
            fmt = schema["format"]
            if fmt in FORMAT_VALIDATORS:
                if not FORMAT_VALIDATORS[fmt](instance):
                    msg = f"string does not match format '{fmt}'"
                    if ctx.strict_formats:
                        if not ctx.add_error(path, msg):
                            return False
                    else:
                        logger.debug(f"{path}: {msg}")
    
    # Number-specific validations
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            if not ctx.add_error(path, f"value {instance} < minimum {schema['minimum']}"):
                return False
        
        if "maximum" in schema and instance > schema["maximum"]:
            if not ctx.add_error(path, f"value {instance} > maximum {schema['maximum']}"):
                return False
        
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            if not ctx.add_error(path, f"value {instance} <= exclusiveMinimum {schema['exclusiveMinimum']}"):
                return False
        
        if "exclusiveMaximum" in schema and instance >= schema["exclusiveMaximum"]:
            if not ctx.add_error(path, f"value {instance} >= exclusiveMaximum {schema['exclusiveMaximum']}"):
                return False
        
        if "multipleOf" in schema:
            divisor = schema["multipleOf"]
            # Use modulo with tolerance for floating point
            remainder = instance % divisor
            if remainder > 1e-10 and abs(remainder - divisor) > 1e-10:
                if not ctx.add_error(path, f"value {instance} is not a multiple of {divisor}"):
                    return False
    
    # Array-specific validations
    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            if not ctx.add_error(path, f"array length {len(instance)} < minItems {schema['minItems']}"):
                return False
        
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            if not ctx.add_error(path, f"array length {len(instance)} > maxItems {schema['maxItems']}"):
                return False
        
        if schema.get("uniqueItems"):
            seen: Set[str] = set()
            for i, item in enumerate(instance):
                # Use JSON serialization for comparison
                key = json.dumps(item, sort_keys=True)
                if key in seen:
                    if not ctx.add_error(path, f"array items are not unique (duplicate at index {i})"):
                        return False
                    break
                seen.add(key)
        
        if "contains" in schema:
            contains_valid = False
            for item in instance:
                test_ctx = ValidationContext(
                    root_schema=ctx.root_schema,
                    fail_fast=True,
                    strict_formats=ctx.strict_formats
                )
                if validate_schema(item, schema["contains"], test_ctx, path) and not test_ctx.errors:
                    contains_valid = True
                    break
            if not contains_valid:
                if not ctx.add_error(path, "array does not contain any item matching 'contains' schema"):
                    return False
        
        # Validate items
        if "items" in schema:
            items_schema = schema["items"]
            if isinstance(items_schema, list):
                # Tuple validation
                for i, item in enumerate(instance):
                    item_path = f"{path}/{i}"
                    if i < len(items_schema):
                        if not validate_schema(item, items_schema[i], ctx, item_path):
                            return False
                    elif "additionalItems" in schema:
                        if schema["additionalItems"] is False:
                            if not ctx.add_error(item_path, "additional items not allowed"):
                                return False
                        elif isinstance(schema["additionalItems"], dict):
                            if not validate_schema(item, schema["additionalItems"], ctx, item_path):
                                return False
            else:
                # All items must match
                for i, item in enumerate(instance):
                    item_path = f"{path}/{i}"
                    if not validate_schema(item, items_schema, ctx, item_path):
                        return False
    
    # Object-specific validations
    if isinstance(instance, dict):
        # Required properties
        for key in schema.get("required", []):
            if key not in instance:
                if not ctx.add_error(path, f"missing required property '{key}'"):
                    return False
        
        if "minProperties" in schema and len(instance) < schema["minProperties"]:
            if not ctx.add_error(path, f"object has {len(instance)} properties < minProperties {schema['minProperties']}"):
                return False
        
        if "maxProperties" in schema and len(instance) > schema["maxProperties"]:
            if not ctx.add_error(path, f"object has {len(instance)} properties > maxProperties {schema['maxProperties']}"):
                return False
        
        # Property dependencies
        if "dependencies" in schema:
            for prop, dep in schema["dependencies"].items():
                if prop in instance:
                    if isinstance(dep, list):
                        # Property dependency
                        for required in dep:
                            if required not in instance:
                                if not ctx.add_error(path, f"property '{prop}' requires '{required}'"):
                                    return False
                    else:
                        # Schema dependency
                        if not validate_schema(instance, dep, ctx, path):
                            return False
        
        # Validate properties
        properties = schema.get("properties", {})
        pattern_properties = schema.get("patternProperties", {})
        additional = schema.get("additionalProperties", True)
        
        for key, value in instance.items():
            prop_path = f"{path}/{key}"
            validated = False
            
            # Check properties
            if key in properties:
                validated = True
                if not validate_schema(value, properties[key], ctx, prop_path):
                    return False
            
            # Check patternProperties
            for pattern, prop_schema in pattern_properties.items():
                if re.search(pattern, key):
                    validated = True
                    if not validate_schema(value, prop_schema, ctx, prop_path):
                        return False
            
            # Check additionalProperties
            if not validated:
                if additional is False:
                    if not ctx.add_error(prop_path, "additional property not allowed"):
                        return False
                elif isinstance(additional, dict):
                    if not validate_schema(value, additional, ctx, prop_path):
                        return False
        
        # Property names
        if "propertyNames" in schema:
            for key in instance.keys():
                if not validate_schema(key, schema["propertyNames"], ctx, f"{path}/{key}#key"):
                    return False
    
    # Conditional validation (if/then/else)
    if "if" in schema:
        test_ctx = ValidationContext(
            root_schema=ctx.root_schema,
            fail_fast=True,
            strict_formats=ctx.strict_formats
        )
        if_valid = validate_schema(instance, schema["if"], test_ctx, path) and not test_ctx.errors
        
        if if_valid and "then" in schema:
            if not validate_schema(instance, schema["then"], ctx, path):
                return False
        elif not if_valid and "else" in schema:
            if not validate_schema(instance, schema["else"], ctx, path):
                return False
    
    return True


def validate(
    instance: JsonType,
    schema: JsonObject,
    fail_fast: bool = False,
    strict_formats: bool = False
) -> ValidationErrors:
    """
    Validate an instance against a JSON Schema.
    
    Args:
        instance: The value to validate
        schema: The JSON Schema
        fail_fast: Stop on first error if True
        strict_formats: Treat format mismatches as errors if True
    
    Returns:
        List of validation error messages
    """
    ctx = ValidationContext(
        root_schema=schema,
        fail_fast=fail_fast,
        strict_formats=strict_formats
    )
    validate_schema(instance, schema, ctx, "")
    return sorted(ctx.errors)  # Sort for deterministic output


# =============================================================================
# Schema Utilities
# =============================================================================

def apply_defaults(instance: JsonType, schema: JsonObject, root_schema: Optional[JsonObject] = None) -> JsonType:
    """
    Apply default values from schema to instance.
    
    Args:
        instance: The value to augment
        schema: The JSON Schema
        root_schema: Root schema for $ref resolution
    
    Returns:
        Instance with defaults applied
    """
    if root_schema is None:
        root_schema = schema
    
    # Handle $ref
    if "$ref" in schema:
        ref = schema["$ref"]
        if ref.startswith("#"):
            found, ref_schema = get_by_pointer(root_schema, ref[1:])
            if found:
                return apply_defaults(instance, ref_schema, root_schema)
        return instance
    
    # Apply default if instance is None
    if instance is None and "default" in schema:
        return copy.deepcopy(schema["default"])
    
    # Recurse into objects
    if isinstance(instance, dict):
        result = copy.copy(instance)
        properties = schema.get("properties", {})
        
        for key, prop_schema in properties.items():
            if key in result:
                result[key] = apply_defaults(result[key], prop_schema, root_schema)
            elif "default" in prop_schema:
                result[key] = copy.deepcopy(prop_schema["default"])
        
        return result
    
    # Recurse into arrays
    if isinstance(instance, list):
        items_schema = schema.get("items")
        if items_schema and isinstance(items_schema, dict):
            return [apply_defaults(item, items_schema, root_schema) for item in instance]
    
    return instance


# =============================================================================
# Filtering Operations
# =============================================================================

@dataclass
class FilterConfig:
    """Configuration for filtering operations."""
    include: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)
    rename: Dict[str, str] = field(default_factory=dict)  # old_path -> new_path
    compact_arrays: bool = True
    
    @classmethod
    def from_dict(cls, data: JsonObject) -> "FilterConfig":
        """Create FilterConfig from a dictionary."""
        return cls(
            include=data.get("include", []),
            exclude=data.get("exclude", []),
            rename=data.get("rename", {}),
            compact_arrays=data.get("compactArrays", True)
        )


def build_from_includes(doc: JsonType, includes: List[str]) -> JsonType:
    """
    Build a new document containing only the specified paths.
    
    Args:
        doc: Source document
        includes: List of JSON pointers (supports wildcards)
    
    Returns:
        New document with only included paths
    """
    output: JsonType = {} if isinstance(doc, dict) else [] if isinstance(doc, list) else None
    
    for pattern in includes:
        for pointer in expand_wildcards(doc, pattern):
            found, value = get_by_pointer(doc, pointer)
            if found:
                output = set_by_pointer(output, pointer, copy.deepcopy(value))
    
    return output


def apply_filters(doc: JsonType, config: FilterConfig) -> JsonType:
    """
    Apply filtering operations to a document.
    
    Operations are applied in order:
    1. Include (if specified, builds sparse document)
    2. Exclude (removes paths)
    3. Rename (moves values to new paths)
    
    Args:
        doc: Source document
        config: Filter configuration
    
    Returns:
        Filtered document
    """
    # Step 1: Include
    if config.include:
        output = build_from_includes(doc, config.include)
    else:
        output = copy.deepcopy(doc)
    
    # Step 2: Exclude
    for pattern in config.exclude:
        for pointer in list(expand_wildcards(output, pattern)):
            delete_by_pointer(output, pointer, config.compact_arrays)
    
    # Step 3: Rename
    for old_path, new_path in config.rename.items():
        for pointer in list(expand_wildcards(output, old_path)):
            found, value = get_by_pointer(output, pointer)
            if found:
                delete_by_pointer(output, pointer, config.compact_arrays)
                # Compute the new pointer (replace the matched pattern portion)
                if "*" in old_path:
                    # For wildcards, map the concrete path to the new pattern
                    old_parts = decode_pointer(old_path)
                    new_parts = decode_pointer(new_path)
                    concrete_parts = decode_pointer(pointer)
                    
                    result_parts = []
                    concrete_idx = 0
                    for i, (old_p, new_p) in enumerate(zip(old_parts, new_parts)):
                        if old_p in ("*", "**"):
                            result_parts.append(concrete_parts[concrete_idx])
                        else:
                            result_parts.append(new_p)
                        concrete_idx += 1
                    
                    new_pointer = encode_pointer(result_parts)
                else:
                    new_pointer = new_path
                
                output = set_by_pointer(output, new_pointer, value)
    
    return output


def compact_nulls(doc: JsonType) -> JsonType:
    """
    Remove null values from arrays recursively.
    
    Args:
        doc: Document to compact
    
    Returns:
        Document with nulls removed from arrays
    """
    if isinstance(doc, dict):
        return {k: compact_nulls(v) for k, v in doc.items()}
    elif isinstance(doc, list):
        return [compact_nulls(item) for item in doc if item is not None]
    return doc


# =============================================================================
# Output Formatting
# =============================================================================

def canonical_json(obj: JsonType) -> str:
    """
    Produce canonical JSON output.
    
    Canonical form:
    - Sorted keys
    - No unnecessary whitespace
    - Consistent float formatting
    - ASCII-only output
    
    Args:
        obj: JSON value to serialize
    
    Returns:
        Canonical JSON string
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False
    )


def pretty_json(obj: JsonType, indent: int = 2) -> str:
    """
    Produce pretty-printed JSON output.
    
    Args:
        obj: JSON value to serialize
        indent: Indentation level
    
    Returns:
        Pretty-printed JSON string
    """
    return json.dumps(
        obj,
        sort_keys=True,
        indent=indent,
        ensure_ascii=True
    )


def compute_hash(data: str) -> str:
    """Compute SHA256 hash of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# I/O Operations
# =============================================================================

def load_json(path: str) -> JsonType:
    """
    Load JSON from a file or stdin.
    
    Args:
        path: File path or "-" for stdin
    
    Returns:
        Parsed JSON value
    
    Raises:
        IOError: If file cannot be read
        json.JSONDecodeError: If JSON is invalid
    """
    try:
        if path == "-":
            return json.load(sys.stdin)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise IOError(f"File not found: {path}")
    except PermissionError:
        raise IOError(f"Permission denied: {path}")


def save_json(data: str, path: str) -> None:
    """
    Save JSON string to a file or stdout.
    
    Args:
        data: JSON string to write
        path: File path or "-" for stdout
    """
    if path == "-":
        sys.stdout.write(data)
        sys.stdout.write("\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.write(data)
            f.write("\n")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Deterministic JSON transformer with schema validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input data.json --schema schema.json --pretty
  echo '{"id": "1", "name": "test"}' | %(prog)s --schema schema.json
  %(prog)s --input data.json --filter filter.json --output result.json
  %(prog)s --input data.json --validate-only --schema schema.json

Filter file format (filter.json):
  {
    "include": ["/id", "/name", "/items/*/value"],
    "exclude": ["/secret", "/internal/**"],
    "rename": {"/old_field": "/new_field"},
    "compactArrays": true
  }
        """
    )
    
    # Input/Output
    parser.add_argument(
        "--input", "-i",
        default="-",
        help="Input JSON file or '-' for stdin (default: stdin)"
    )
    parser.add_argument(
        "--output", "-o",
        default="-",
        help="Output file or '-' for stdout (default: stdout)"
    )
    
    # Schema and Filter
    parser.add_argument(
        "--schema", "-s",
        help="JSON Schema file for validation"
    )
    parser.add_argument(
        "--filter", "-f",
        dest="filter_path",
        help="Filter configuration file"
    )
    
    # Output format
    format_group = parser.add_mutually_exclusive_group()
    format_group.add_argument(
        "--pretty", "-p",
        action="store_true",
        help="Pretty-print output with indentation"
    )
    format_group.add_argument(
        "--canonical", "-c",
        action="store_true",
        help="Produce canonical JSON (minimal, sorted)"
    )
    
    # Validation options
    parser.add_argument(
        "--validate-only", "-V",
        action="store_true",
        help="Only validate, don't transform or output"
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop validation on first error"
    )
    parser.add_argument(
        "--strict-formats",
        action="store_true",
        help="Treat format validation failures as errors"
    )
    parser.add_argument(
        "--apply-defaults",
        action="store_true",
        help="Apply default values from schema"
    )
    
    # Misc options
    parser.add_argument(
        "--hash",
        action="store_true",
        help="Print SHA256 hash of output to stderr"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-error output"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase verbosity (can be repeated)"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 2.0.0"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose >= 2:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose >= 1:
        logging.getLogger().setLevel(logging.INFO)
    
    # Load input
    try:
        payload = load_json(args.input)
        logger.info(f"Loaded input from {args.input}")
    except IOError as e:
        logger.error(str(e))
        return ExitCode.IO_ERROR
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in input: {e}")
        return ExitCode.IO_ERROR
    
    # Load and apply schema
    schema = None
    if args.schema:
        try:
            schema = load_json(args.schema)
            logger.info(f"Loaded schema from {args.schema}")
        except IOError as e:
            logger.error(str(e))
            return ExitCode.IO_ERROR
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in schema: {e}")
            return ExitCode.SCHEMA_ERROR
        
        # Validate
        errors = validate(
            payload,
            schema,
            fail_fast=args.fail_fast,
            strict_formats=args.strict_formats
        )
        
        if errors:
            for err in errors:
                print(f"Validation error: {err}", file=sys.stderr)
            return ExitCode.VALIDATION_ERROR
        
        logger.info("Validation passed")
        
        # Apply defaults
        if args.apply_defaults:
            payload = apply_defaults(payload, schema)
            logger.info("Applied defaults from schema")
    
    # Validate-only mode
    if args.validate_only:
        if not args.quiet:
            print("Validation successful", file=sys.stderr)
        return ExitCode.SUCCESS
    
    # Load and apply filters
    if args.filter_path:
        try:
            filter_data = load_json(args.filter_path)
            filter_config = FilterConfig.from_dict(filter_data)
            logger.info(f"Loaded filter from {args.filter_path}")
        except IOError as e:
            logger.error(str(e))
            return ExitCode.IO_ERROR
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in filter: {e}")
            return ExitCode.FILTER_ERROR
        
        try:
            payload = apply_filters(payload, filter_config)
            if filter_config.compact_arrays:
                payload = compact_nulls(payload)
            logger.info("Applied filters")
        except (PointerError, FilterError) as e:
            logger.error(f"Filter error: {e}")
            return ExitCode.FILTER_ERROR
    
    # Format output
    if args.canonical:
        output = canonical_json(payload)
    elif args.pretty:
        output = pretty_json(payload)
    else:
        output = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    
    # Compute hash
    if args.hash:
        h = compute_hash(output)
        print(f"SHA256: {h}", file=sys.stderr)
    
    # Write output
    try:
        save_json(output, args.output)
        logger.info(f"Wrote output to {args.output}")
    except IOError as e:
        logger.error(str(e))
        return ExitCode.IO_ERROR
    
    return ExitCode.SUCCESS


if __name__ == "__main__":
    raise SystemExit(main())
