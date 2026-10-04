"""Data-only contract proposals; isolated learning and postcondition acceptance."""
import json
import math
import re

from vectorpro.host import HOST_TYPES


def validate_draft(data):
    required = {"version", "name", "description", "parameters", "output", "allowed_operations"}
    if not isinstance(data, dict) or set(data) != required:
        raise ValueError("draft needs only version/name/description/parameters/output/allowed_operations; no code or execution tensors")
    if type(data["version"]) is not int or data["version"] != 1:
        raise ValueError("unsupported draft version")
    for field, maximum in (("name",64),("description",2000)):
        if not isinstance(data[field], str) or not 1 <= len(data[field]) <= maximum:
            raise ValueError(f"draft {field} must be a nonempty string up to {maximum} characters")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}",data["name"]):
        raise ValueError("draft name must be an identifier")
    parameters = data["parameters"]
    if not isinstance(parameters,list) or not 1 <= len(parameters) <= 3:
        raise ValueError("draft needs 1..3 parameters")
    names=[]
    for p in parameters:
        if not isinstance(p,dict) or set(p)!={"name","type","role"}:
            raise ValueError("each parameter needs only name/type/role")
        if (not isinstance(p["name"],str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}",p["name"])
                or p["name"] in ("width","version","contract_id")):
            raise ValueError("parameter name must be a nonreserved identifier")
        if p["type"] not in ("value","path","buffer"):
            raise ValueError("parameter type must be value/path/buffer")
        if not isinstance(p["role"],str) or not 1 <= len(p["role"]) <= 500:
            raise ValueError("parameter role must be a nonempty string up to 500 characters")
        names.append(p["name"])
    if len(set(names)) != len(names):
        raise ValueError("parameter names must be unique")
    output=data["output"]
    if (not isinstance(output,dict) or set(output)!={"type","width"}
            or output["type"] not in ("value","path","buffer") or output["width"] not in ("W","W+1","2W","1")):
        raise ValueError("output needs a portable type and W/W+1/2W/1 width")
    operations=data["allowed_operations"]
    if (not isinstance(operations,list) or any(not isinstance(o,str) or o not in HOST_TYPES for o in operations)
            or len(set(operations))!=len(operations)):
        raise ValueError("allowed_operations must be distinct provided host operation names")
    return json.loads(json.dumps(data))


def numeric_lesson(data):
    from vectorpro.learning.examples import ExampleLesson
    if not isinstance(data,dict) or set(data)!={"training","validation"}:
        raise ValueError("numeric evidence needs only training/validation")
    for split,minimum in (("training",4),("validation",2)):
        row=data[split]
        if not isinstance(row,dict) or set(row)!={"width","operands","targets"}:
            raise ValueError("numeric examples need only width/operands/targets")
        if type(row["width"]) is not int or not 1<=row["width"]<=32:
            raise ValueError("numeric evidence width must be 1..32")
        if not isinstance(row["operands"],list) or not minimum<=len(row["operands"])<=256:
            raise ValueError("provide 4..256 training and 2..256 validation inputs")
        if not isinstance(row["targets"],list) or len(row["targets"])!=len(row["operands"]):
            raise ValueError("each example needs a target")
        if any(not isinstance(r,list) for r in row["operands"]):
            raise ValueError("operand rows must be arrays")
    return ExampleLesson.from_dict(data)


def state_lesson(data):
    from vectorpro.learning.stateful import StateLesson
    allowed={"input_types","training","validation","width","max_steps","candidate_budget",
             "control_flow","time_budget_seconds","buffer_loops","execution_budget","list_loops","list_reduction"}
    if not isinstance(data,dict) or set(data)-allowed or not {"input_types","training","validation"}<=set(data):
        raise ValueError("state evidence accepts only types/examples and bounded search settings")
    for field,minimum,maximum in (("width",1,32),("max_steps",1,12),("candidate_budget",1,20000),
                                   ("execution_budget",1,20000)):
        if field in data and (type(data[field]) is not int or not minimum<=data[field]<=maximum):
            raise ValueError(f"state {field} exceeds learning limits")
    budget=data.get("time_budget_seconds",60)
    if type(budget) not in (int,float) or not math.isfinite(budget) or not 0<budget<=60:
        raise ValueError("state time budget must be finite and in (0,60]")
    for split in ("training","validation"):
        if not isinstance(data[split],list) or not 1<=len(data[split])<=8:
            raise ValueError("state evidence needs 1..8 cases per split")
        for case in data[split]:
            if (not isinstance(case,dict) or set(case)-{"inputs","before","after","output","operations","before_directories","after_directories"}
                    or not {"inputs","before","after"}<=set(case)):
                raise ValueError("state cases accept only inputs/snapshots/output/operation observations")
            if not isinstance(case["inputs"], list) or not 1 <= len(case["inputs"]) <= 3:
                raise ValueError("state cases need 1..3 inputs")
            buffers = [v for v, kind in zip(case["inputs"], data["input_types"]) if kind == "buffer"]
            if any(not isinstance(v, str) for v in buffers) or sum(len(v) for v in buffers) > 131072:
                raise ValueError("buffer evidence exceeds input byte limit")
            for field in ("before_directories", "after_directories"):
                if field in case and (not isinstance(case[field], list) or len(case[field]) > 32
                                     or any(not isinstance(p, str) for p in case[field])):
                    raise ValueError("state directory snapshots need at most 32 paths")
            for field in ("before","after"):
                files=case[field]
                if not isinstance(files,dict) or len(files)>32:
                    raise ValueError("state snapshots need at most 32 files")
                if any(not isinstance(v,str) or not re.fullmatch(r"(?:[0-9a-fA-F]{2})*",v) for v in files.values()):
                    raise ValueError("state snapshots require hexadecimal byte pairs")
                if sum(len(v) for v in files.values())>131072:
                    raise ValueError("state snapshot exceeds evidence byte limit")
    lesson=StateLesson.from_dict(data)
    lesson.validate()
    return lesson
