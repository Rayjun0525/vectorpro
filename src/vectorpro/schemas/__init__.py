from vectorpro.schemas.base import Schema
from vectorpro.schemas.map import MapSchema
from vectorpro.schemas.scan import Direction, ScanSchema

_KINDS = {"map": MapSchema, "scan": ScanSchema}


def schema_from_spec(spec: dict) -> Schema:
    return _KINDS[spec["kind"]].from_spec(spec)


__all__ = ["Direction", "MapSchema", "ScanSchema", "Schema", "schema_from_spec"]
