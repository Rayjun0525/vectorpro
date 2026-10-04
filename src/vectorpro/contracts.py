"""Versioned, tensor-derived function contracts shared by callers and adapters.

Content IDs identify a specific stored implementation and dependency closure.
They are not semantic equivalence IDs or a proof of correctness.
"""
import hashlib
import json

from vectorpro.host import HOST_TYPES
from vectorpro.machine import VectorProgram

VERSION = 1


def _closure(registry, name, active=()):
    if name in active:
        raise ValueError("recursive capabilities cannot export a finite contract")
    cap = registry.get(name)
    dependencies = []
    if cap.provenance["kind"] == "program":
        program = VectorProgram.from_data(cap.provenance["program"])
        names = sorted({registry.name_of(program.keys[i]) for i in range(program.n_steps)
                        if bool(program.calls[i])})
        dependencies = [_closure(registry, n, (*active, name)) for n in names]
    return {"name": name, "arity": cap.plan.arity, "output_width": cap.plan.output.value,
            "address": cap.key.tolist(), "provenance": cap.provenance, "dependencies": dependencies}


def _operations(closure):
    operations = []
    if closure["provenance"]["kind"] == "host":
        operations.append(closure["provenance"]["operation"])
    for child in closure["dependencies"]:
        operations.extend(_operations(child))
    return sorted(set(operations))


def build_contract(registry, name):
    # Imported lazily: the semantic catalog also consumes this common contract.
    from vectorpro.semantic_catalog import argument_roles, document, input_types
    cap = registry.get(name)
    kinds = input_types(cap)
    if len(kinds) != cap.plan.arity or any(k not in ("value", "path", "buffer") for k in kinds):
        raise ValueError("capability has an invalid input type contract")
    output = (HOST_TYPES[cap.provenance["operation"]][1] if cap.provenance["kind"] == "host"
              else cap.provenance.get("output_type", "unknown" if cap.executable.effects else "value"))
    closure = _closure(registry, name)
    encoded = json.dumps(closure, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    parameters = []
    for i, (kind, role) in enumerate(zip(kinds, argument_roles(cap))):
        schema = {"type": "integer", "minimum": 0} if kind == "value" else {"type": "string"}
        if kind == "buffer":
            schema["pattern"] = "^(?:[0-9a-fA-F]{2})*$"
        parameters.append({"name": f"x{i}", "type": kind, "role": role, "schema": schema})
    contract = {"version": VERSION, "name": name, "description": document(registry, name),
                "input_types": kinds, "argument_roles": argument_roles(cap),
                "output_width": cap.plan.output.value, "has_effects": bool(cap.executable.effects),
                "parameters": parameters, "output": {"type": output, "width": cap.plan.output.value},
                "width": {"minimum": 1, "maximum": 64},
                "execution": {"address": cap.key.tolist(), "requires_host": bool(cap.executable.effects),
                              "operations": _operations(closure), "digest": hashlib.sha256(encoded).hexdigest()},
                "verification": {"source": "stored training/validation records; not independent intent proof",
                                 "history": cap.history, "audit": cap.audit}}
    if "contract_draft" in cap.provenance:
        from vectorpro.contract_learning import validate_draft
        draft = validate_draft(cap.provenance["contract_draft"])
        if ([p["type"] for p in draft["parameters"]] != kinds or draft["output"] != contract["output"]
                or draft["name"] != name or set(contract["execution"]["operations"])-set(draft["allowed_operations"])):
            raise ValueError("stored draft does not match the acquired implementation")
        for parameter, declared in zip(parameters,draft["parameters"]):
            parameter.update(name=declared["name"],role=declared["role"],role_source="draft declaration, not independent proof")
        contract["interface_origin"] = "validated draft shape/effect bounds; descriptive roles are supplied"
    identity = json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    contract["id"] = "vp1:" + hashlib.sha256(identity).hexdigest()
    return json.loads(json.dumps(contract))  # detached data: callers cannot mutate capability metadata


def tool_schema(contract):
    arguments = {p["name"]: {**p["schema"], "description": p["role"]} for p in contract["parameters"]}
    return {"type": "function", "function": {
        "name": "vp_" + contract["id"].split(":")[1][:60],
        "description": contract["description"] + " Contract ID: " + contract["id"],
        "parameters": {"type": "object", "properties": {
            "width": {"type": "integer", **contract["width"]},
            "arguments": {"type": "object", "properties": arguments,
                          "required": list(arguments), "additionalProperties": False}},
            "required": ["width", "arguments"], "additionalProperties": False}}}
