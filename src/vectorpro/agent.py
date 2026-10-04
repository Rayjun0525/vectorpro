"""Optional function-calling LLM adapter. Acquired programs run without an LLM."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Protocol
from urllib.request import Request, urlopen

from vectorpro.host import HostContext
from vectorpro.learning import ExampleLesson, LearningPlan
from vectorpro.learning.stateful import StateLesson
from vectorpro.runtime import VectorRuntime


GUIDE = """You prepare evidence and use a learnable vector runtime, not a code interpreter.
Call ONE tool at a time. Tools are separate functions: list_capabilities is NOT a
capability name for execute. After an error, correct the arguments and retry.
For an ambiguous request ALWAYS call ask_user, rather than asking in plain text.
Do not invent XML tags such as <ask_user>. Call ask_user through the tool protocol.
For numeric learning prefer teach_numeric: supply your own training and validation
examples; the adapter derives the learning schedule, never the answers or algorithm.
Provide at least 4 training examples and 2 validation examples for teach_numeric.
Never write Python-like calls in text. Use the actual function-calling protocol.
The execute operands JSON is [[12,3]] for numbers and [[{"utf8":"input.bin"},53]]
for a path and a number. Both examples show argument format only.
Look up the actual capability name and its arity before executing.
Read list_capabilities and describe_capability before teaching. Known functions execute
directly. Unknown functions need independent training/validation example sets. Use teach
to acquire a function in memory, then execute only the user's requested native operation.
Missing/ambiguous goals: ask_user; do not invent the user's desired output or claim success.
No source code, binaries, instruction recipes or tensor programs may be supplied.
Numeric teaching: {name, plan: {name,description,arity,output: 'W'|'W+1'|'2W'|'1',
rounds:[count], train_width,validation_width,validation_examples}, lesson:
{training:{width,operands:[[int]],targets:[int]},validation:{width,operands,targets}}}.
State teaching: {name,state_lesson:{input_types:['path'|'value'], training:[{inputs,
before:{relative_path:hex},after:{relative_path:hex},output?:int,operations?:[host_name]}],
validation:[same shape],width:16,max_steps:7,candidate_budget:5000,
control_flow:true,buffer_loops:true,time_budget_seconds:60,execution_budget:20000}}.
before/after are complete file snapshots; unchanged files must remain in after.
buffer_loops searches indexed fill/map grammar; acquired numeric stepping is needed.
max_steps counts calls. Validation is separate; model-generated examples are evidence
proposals, not independent proof of the human's intent or general correctness.
execute: {name, width, operands:[[integer|{utf8:relative_path}|{hex:bytes_hex}]]}.
Only supplied host-root files are accessible. ask_user takes {question:string}.
Trust tool statuses; learned means checked examples, learning_failed means no solution.
"""


def tool(name, description, properties, required=()):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": list(required),
                           "additionalProperties": False}}}


EXAMPLE_SCHEMA = {"type": "object", "properties": {
    "width": {"type": "integer", "minimum": 1, "maximum": 32},
    "operands": {"type": "array", "minItems": 1, "maxItems": 256,
                 "items": {"type": "array", "minItems": 1, "maxItems": 3, "items": {"type": "integer"}}},
    "targets": {"type": "array", "minItems": 1, "maxItems": 256, "items": {"type": "integer"}}},
    "required": ["width", "operands", "targets"], "additionalProperties": False}


TOOLS = [tool("list_capabilities", "List acquired and provided capabilities with types", {}),
         tool("teach_numeric", "Acquire a NEW numeric function from YOUR examples. No code. Separate training and validation.",
              {"name": {"type": "string"}, "description": {"type": "string"},
               "output": {"type": "string", "enum": ["W", "W+1", "2W", "1"]},
               "training": {**EXAMPLE_SCHEMA, "properties": {**EXAMPLE_SCHEMA["properties"],
                   "operands": {**EXAMPLE_SCHEMA["properties"]["operands"], "minItems": 4},
                   "targets": {**EXAMPLE_SCHEMA["properties"]["targets"], "minItems": 4}}},
               "validation": {**EXAMPLE_SCHEMA, "properties": {**EXAMPLE_SCHEMA["properties"],
                   "operands": {**EXAMPLE_SCHEMA["properties"]["operands"], "minItems": 2},
                   "targets": {**EXAMPLE_SCHEMA["properties"]["targets"], "minItems": 2}}}},
              ("name", "description", "output", "training", "validation")),
         tool("describe_capability", "Read a capability's actual learned behaviour", {"name": {"type": "string"}}, ("name",)),
         tool("teach", "Learn from a data-only numeric/state lesson; never executes on native files",
              {"request": {"type": "object"}}, ("request",)),
         tool("execute", "Execute a known capability requested by the user",
              {"name": {"type": "string"}, "width": {"type": "integer"},
               "operands": {"type": "array", "items": {"type": "array", "items": {"oneOf": [
                   {"type": "integer"}, {"type": "object", "properties": {"utf8": {"type": "string"}}, "required": ["utf8"], "additionalProperties": False},
                   {"type": "object", "properties": {"hex": {"type": "string"}}, "required": ["hex"], "additionalProperties": False}]}}}},
              ("name", "width", "operands")),
         tool("ask_user", "Ask for missing intent/evidence and stop this turn", {"question": {"type": "string"}}, ("question",))]

_EXECUTE_PROPERTIES = next(t["function"]["parameters"]["properties"] for t in TOOLS if t["function"]["name"] == "execute")
RESOLVE_TOOL = tool("resolve_request", "Find typed candidates and verify supplied evidence in memory; NEVER executes on native files",
    {"query": {"type": "string"}, "width": _EXECUTE_PROPERTIES["width"],
     "operands": _EXECUTE_PROPERTIES["operands"], "numeric_validation": EXAMPLE_SCHEMA,
     "state_validation": {"type": "array", "minItems": 1, "maxItems": 8,
         "items": {"type": "object", "properties": {
             "inputs": _EXECUTE_PROPERTIES["operands"]["items"],
             "before": {"type": "object", "additionalProperties": {"type": "string", "pattern": "^[0-9a-fA-F]*$"}},
             "after": {"type": "object", "additionalProperties": {"type": "string", "pattern": "^[0-9a-fA-F]*$"}},
             "output": {"type": "integer"}}, "required": ["inputs", "before", "after"], "additionalProperties": False}}},
    ("query", "width", "operands"))

_NUMERIC_ROW = {"type": "array", "minItems": 1, "maxItems": 3, "items": {"type": "integer"}}
RESOLVE_NUMERIC = tool("resolve_numeric", "Find a stored numeric function matching correct examples; no native execution",
    {"query": {"type": "string"}, "width": {"type": "integer", "minimum": 1, "maximum": 32},
     "inputs": _NUMERIC_ROW, "validation_width": {"type": "integer", "minimum": 1, "maximum": 32},
     "validation_inputs": {"type": "array", "minItems": 2, "maxItems": 32, "items": _NUMERIC_ROW},
     "validation_outputs": {"type": "array", "minItems": 2, "maxItems": 32, "items": {"type": "integer"}}},
    ("query", "width", "inputs", "validation_width", "validation_inputs", "validation_outputs"))
RESOLVE_STATE = tool("resolve_state", "Find a stored file/buffer procedure matching complete virtual file snapshots; no native execution",
    {"query": {"type": "string"}, "width": _EXECUTE_PROPERTIES["width"],
     "inputs": _EXECUTE_PROPERTIES["operands"]["items"],
     "validation": RESOLVE_TOOL["function"]["parameters"]["properties"]["state_validation"]},
    ("query", "width", "inputs", "validation"))
EXECUTE_RESOLVED = tool("execute_resolved", "Execute the last uniquely matched request with its already verified inputs, once", {})
PROPOSE_TOOL = tool("propose_request", "Step 1: extract ONLY the actual requested inputs and goal; do not include validation example inputs",
    {"query": {"type": "string", "description": "Operation/goal in words, not an input number"},
     "width": {"type": "integer", "minimum": 1, "maximum": 32},
     "inputs": {**_EXECUTE_PROPERTIES["operands"]["items"], "minItems": 1, "maxItems": 3,
                "description": "All inputs of the actual operation, in order. JSON integers or {utf8:relative_path}."}},
    ("query", "width", "inputs"))
PROPOSE_NUMERIC = tool("propose_numeric_request", "Step 1 for NUMBERS: extract every actual numeric input, not the validation examples",
    {"query": {"type": "string", "description": "Requested operation in words"},
     "width": {"type": "integer", "minimum": 1, "maximum": 32},
     "inputs": {**_NUMERIC_ROW, "description": "All requested numbers in order, as JSON integers. Never strings or file paths."}},
    ("query", "width", "inputs"))
PROPOSE_FILE = tool("propose_file_request", "Step 1 for FILES: extract actual target paths and numeric parameters, not demonstration paths",
    {"query": {"type": "string"}, "width": {"type": "integer", "minimum": 1, "maximum": 32},
     "paths": {"type": "array", "minItems": 1, "maxItems": 3, "items": {"type": "string"}},
     "values": {"type": "array", "minItems": 0, "maxItems": 2, "items": {"type": "integer"},
                "description": "Actual mask/value/flag parameters; empty if none"}},
    ("query", "width", "paths", "values"))
VERIFY_NUMERIC = tool("verify_numeric", "Step 2: provide small correct validation examples ONLY, not the requested large inputs",
    {"width": {"type": "integer", "minimum": 1, "maximum": 32},
     "operands": {"type": "array", "minItems": 2, "maxItems": 32, "items": _NUMERIC_ROW},
     "targets": {"type": "array", "minItems": 2, "maxItems": 32, "items": {"type": "integer"}}},
    ("width", "operands", "targets"))
VERIFY_STATE = tool("verify_state", "Step 2: provide complete before/after file snapshots as HEX byte strings",
    {"cases": RESOLVE_TOOL["function"]["parameters"]["properties"]["state_validation"]}, ("cases",))
SEARCH_GOAL = tool("search_goal", "Find candidate functions and their argument roles for the requested operation; no execution",
                   {"query": {"type": "string"}}, ("query",))

CATALOG_GUIDE = """Use actual tools, one call per turn. For an explicit operation, ALWAYS search_goal first.
Ask the user only if the goal is missing, or verification says more evidence/learning is needed.
Follow the steps:
1. search_goal(query=operation in words).
2. Choose prepare_N using the returned descriptions. Fill every x0,x1,... from the ACTUAL
request, following argument roles, not order of mention. Read the required fields carefully.
Never drop a requested operand to fit a candidate with fewer inputs.
3. Caller validation examples may arrive AFTER prepare_N. They are DIFFERENT inputs;
never change the already fixed request. Use verify_numeric(width,operands,targets) for
two distinct correct examples, or verify_state(cases) for full before/after file snapshots.
State example inputs use integers, {"utf8":"file"} paths or {"hex":"00ff"} buffers.
Snapshots map actual filenames directly to HEX bytes, including unchanged files.
Copy caller evidence accurately; model-generated answers are fallible proposals.
4. On matched, execute_resolved({}) once. Report the actual returned output.
On needs_evidence or needs_learning_examples, ask_user; never substitute another operation.
If learning is enabled and authorized, teaching needs separate training and validation.
On error, correct the failed arguments. No code, binaries, recipes or invented tool names.
"""


class ChatModel(Protocol):
    def complete(self, messages: list[dict], tools: list[dict]) -> dict: ...


class HTTPChatModel:
    """Chat Completions function-call transport; endpoint and model are explicit."""
    def __init__(self, endpoint: str, model: str, api_key: str | None = None, timeout: float = 60):
        self.endpoint, self.model, self.api_key, self.timeout = endpoint, model, api_key, timeout

    def complete(self, messages, tools):
        payload = {"model": self.model, "messages": messages, "tools": tools, "parallel_tool_calls": False}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(self.endpoint, json.dumps(payload).encode("utf-8"), headers, method="POST")
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)["choices"][0]["message"]


class AgentSession:
    def __init__(self, runtime: VectorRuntime, model: ChatModel, program_path: Path | None = None,
                 max_calls: int = 12, catalog=None, allow_learning: bool = True):
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError("max_calls must be a positive integer")
        self.runtime, self.model, self.program_path, self.max_calls = runtime, model, program_path, max_calls
        if catalog is not None and catalog.runtime is not runtime:
            raise ValueError("catalog and session must use the same runtime")
        self.catalog, self._resolved, self._resolution_status = catalog, None, None
        self._proposed = None
        self._goal, self._candidates = None, []
        self.allow_learning = allow_learning
        self.messages = [{"role": "system", "content": CATALOG_GUIDE if catalog else GUIDE}]
        if not allow_learning:
            self.messages[0]["content"] += "\nLearning is disabled in this session; never call teaching tools."

    def available_tools(self):
        if self.catalog is None:
            return [t for t in TOOLS if self.allow_learning or t["function"]["name"] not in ("teach", "teach_numeric")]
        permitted = {"ask_user"}
        if self._resolved or (self._proposed and getattr(self, "_caller_evidence", None)
                              and not getattr(self, "_verification_attempted", False)):
            permitted.clear()  # already supplied evidence/verified inputs need no extra confirmation
        if self._resolution_status == "needs_learning_examples" and self.allow_learning:
            permitted.update(("teach", "teach_numeric"))
        verification = []
        if self._proposed and self._resolution_status not in ("matched", "needs_learning_examples"):
            from vectorpro.semantic_catalog import operand_types
            shape = operand_types(self._proposed["operands"])
            verification = [VERIFY_NUMERIC if all(t == "value" for t in shape) else VERIFY_STATE]
            if verification == [VERIFY_NUMERIC]:
                numeric_tool = json.loads(json.dumps(VERIFY_NUMERIC))
                properties = numeric_tool["function"]["parameters"]["properties"]
                properties["operands"]["items"].update(minItems=len(shape), maxItems=len(shape))
                supplied = getattr(self, "_caller_evidence", None)
                if supplied and "numeric_validation" in supplied:
                    count = len(supplied["numeric_validation"]["operands"])
                    for field in ("operands", "targets"):
                        properties[field].update(minItems=count, maxItems=count)
                verification = [numeric_tool]
            if verification == [VERIFY_STATE]:
                state_tool = json.loads(json.dumps(VERIFY_STATE))
                schemas = []
                for kind in shape:
                    schemas.append({"type": "integer"} if kind == "value" else {
                        "type": "object", "properties": {"utf8" if kind == "path" else "hex": {"type": "string"}},
                        "required": ["utf8" if kind == "path" else "hex"], "additionalProperties": False})
                state_tool["function"]["parameters"]["properties"]["cases"]["items"]["properties"]["inputs"] = {
                    "type": "array", "prefixItems": schemas, "items": False,
                    "minItems": len(shape), "maxItems": len(shape)}
                verification = [state_tool]
        proposal_tools = []
        if not self._proposed:
            if not self._candidates:
                proposal_tools = [SEARCH_GOAL]
            else:
                for index, candidate in enumerate(self._candidates):
                    properties = {"width": {"type": "integer", "minimum": 1, "maximum": 32}}
                    for i, (kind, role) in enumerate(zip(candidate["input_types"], candidate["argument_roles"])):
                        properties[f"x{i}"] = {"type": "integer" if kind == "value" else "string",
                            "description": role + "; actual request input, not validation example"}
                    proposal_tools.append(tool(f"prepare_{index}", "Prepare " + candidate["name"] + ": " + candidate["description"],
                                               properties, tuple(properties)))
        return [*proposal_tools, *verification, *([EXECUTE_RESOLVED] if self._resolved else []),
                *[t for t in TOOLS if t["function"]["name"] in permitted]]

    def call(self, name, args):
        if name in ("teach", "teach_numeric") and not self.allow_learning:
            raise ValueError("learning is disabled for this session")
        if name == "search_goal":
            if self.catalog is None:
                raise ValueError("no tensor catalog configured")
            self._resolved, self._proposed = None, None
            self._goal = args["query"]
            self._candidates = self.catalog.search_goal(self._goal, types=getattr(self, "_caller_types", None))
            self._resolution_status = None if self._candidates else "needs_learning_examples"
            return {"status": "candidates" if self._candidates else "needs_learning_examples", "candidates": self._candidates,
                    "next_step": "choose a prepare_N tool and bind the actual request inputs using its argument roles"}
        if name.startswith("prepare_"):
            index = int(name.removeprefix("prepare_"))
            if not 0 <= index < len(self._candidates):
                raise ValueError("candidate is not available")
            candidate = self._candidates[index]
            if set(args) != {"width", *(f"x{i}" for i in range(len(candidate["input_types"])))}:
                raise ValueError("provide width and every candidate input, without extra fields")
            inputs = []
            for i, kind in enumerate(candidate["input_types"]):
                value = args[f"x{i}"]
                inputs.append({"utf8": value} if kind == "path" else {"hex": value} if kind == "buffer" else value)
            return self.call("propose_request", {"query": self._goal, "width": args["width"], "inputs": inputs})
        if name == "propose_numeric_request":
            from vectorpro.semantic_catalog import operand_types
            if operand_types([args["inputs"]]) != ["value"] * len(args["inputs"]):
                raise ValueError("numeric requests need JSON integer inputs")
            return self.call("propose_request", args)
        if name == "propose_file_request":
            return self.call("propose_request", {"query": args["query"], "width": args["width"],
                "inputs": [{"utf8": p} for p in args["paths"]] + args["values"]})
        if name == "propose_request":
            if self.catalog is None:
                raise ValueError("no tensor catalog configured")
            self._proposed = None
            self._verification_attempted = False
            proposal = {"query": args["query"], "width": args["width"], "operands": [args["inputs"]]}
            result = self.call("resolve_request", proposal)
            self._proposed = json.loads(json.dumps(proposal))
            if result["status"] == "needs_evidence":
                result = {**result, "status": "proposed", "next_step": "verify_numeric or verify_state with the user-provided examples; keep requested inputs separate"}
            return result
        if name in ("verify_numeric", "verify_state"):
            if self._proposed is None:
                raise ValueError("first propose the actual requested inputs")
            self._verification_attempted = True
            key = "numeric_validation" if name == "verify_numeric" else "state_validation"
            supplied = getattr(self, "_caller_evidence", None)
            if supplied is not None and (key not in supplied or supplied[key] != (args if name == "verify_numeric" else args["cases"])):
                raise ValueError("copy caller validation evidence exactly; do not change its inputs, answers or snapshots")
            return self.call("resolve_request", {**self._proposed, key: args if name == "verify_numeric" else args["cases"]})
        if name == "resolve_numeric":
            return self.call("resolve_request", {"query": args["query"], "width": args["width"],
                "operands": [args["inputs"]], "numeric_validation": {"width": args["validation_width"],
                "operands": args["validation_inputs"], "targets": args["validation_outputs"]}})
        if name == "resolve_state":
            return self.call("resolve_request", {"query": args["query"], "width": args["width"],
                "operands": [args["inputs"]], "state_validation": args["validation"]})
        if name == "execute_resolved":
            if args or self._resolved is None:
                raise ValueError("execute_resolved needs no arguments and a uniquely matched request")
            return self.call("execute", dict(self._resolved))
        if name == "resolve_request":
            if self.catalog is None:
                raise ValueError("no tensor catalog configured")
            self._resolved = None
            self._resolution_status = None
            result = self.catalog.resolve(args["query"], args["operands"], args["width"],
                        args.get("numeric_validation"), args.get("state_validation"))
            self._resolution_status = result["status"]
            if result["status"] == "matched":
                self._resolved = {"name": result["name"], "width": args["width"],
                                  "operands": json.loads(json.dumps(args["operands"]))}
            return result
        if name == "teach_numeric":
            if set(args) != {"name", "description", "output", "training", "validation"}:
                raise ValueError("teach_numeric requires name, description, output, training, validation")
            for split in ("training", "validation"):
                examples = args[split]
                if (type(examples["width"]) is not int or not 1 <= examples["width"] <= 32
                        or not (4 if split == "training" else 2) <= len(examples["operands"]) <= 256
                        or len(examples["operands"]) != len(examples["targets"])):
                    raise ValueError("need width 1..32, at least 4 training and 2 validation examples, with matching operands/targets (max 256)")
            arity = len(args["training"]["operands"][0])
            if not 1 <= arity <= 3 or any(len(row) != arity for split in ("training", "validation") for row in args[split]["operands"]):
                raise ValueError("all operand rows must have the same arity, 1..3")
            return self.call("teach", {"request": {"name": args["name"], "plan": {
                "name": args["name"], "description": args["description"], "arity": arity,
                "output": args["output"], "rounds": [len(args["training"]["operands"])],
                "train_width": args["training"]["width"], "validation_width": args["validation"]["width"],
                "validation_examples": len(args["validation"]["operands"])},
                "lesson": {split: args[split] for split in ("training", "validation")}}})
        if name == "list_capabilities":
            return [{"name": c.name, "arity": c.plan.arity, "provided": c.provenance["kind"] == "host",
                     "input_types": c.provenance.get("input_types"), "output_type": c.provenance.get("output_type")}
                    for c in self.runtime.registry]
        if name == "describe_capability":
            return {"description": self.runtime.registry.explain(args["name"])}
        if name == "ask_user":
            return {"status": "needs_input", "question": args["question"]}
        if name == "teach":
            data = args["request"]
            allowed = {"name", "plan", "lesson", "state_lesson"}
            if set(data) - allowed:
                raise ValueError("teaching accepts only plans and examples")
            if "plan" in data:
                p = data["plan"]
                examples = data.get("lesson", {})
                if (p["arity"] > 3 or p.get("train_width", 4) > 32 or p.get("validation_width", 8) > 32
                        or max(p.get("rounds", [8, 16, 32, 64])) > 256
                        or any(len(examples.get(split, {}).get("operands", [])) > 256 for split in ("training", "validation"))):
                    raise ValueError("numeric lesson exceeds agent learning limits")
            state = StateLesson.from_dict(data["state_lesson"]) if "state_lesson" in data else None
            if state and (state.candidate_budget > 20000 or state.time_budget_seconds > 60
                          or (state.execution_budget or 1) > 20000 or state.max_steps > 8):
                raise ValueError("state lesson exceeds agent learning limits")
            result = self.runtime.teach(data["name"],
                        plan=LearningPlan.from_dict(data["plan"]) if "plan" in data else None,
                        lesson=ExampleLesson.from_dict(data["lesson"]) if "lesson" in data else None,
                        state_lesson=state)
            if result.status == "learned":
                self._resolved, self._resolution_status = None, None
                self.runtime.registry.get(data["name"]).provenance["evidence_source"] = "LLM adapter supplied examples"
                self.save()
            return asdict(result)
        if name == "execute":
            if self.catalog is not None:
                if self._resolved is None or args != self._resolved:
                    raise ValueError("resolve_request must verify this exact name/width/operands before native execution")
                self._resolved = None  # a verified request authorizes one execution only
            capability = self.runtime.registry.get(args["name"]) if args["name"] in self.runtime.registry else None
            if not args["operands"]:
                raise ValueError("execute requires non-empty operands, e.g. [[12,3]]; list_capabilities is a separate tool")
            if any(isinstance(value, str) for row in args["operands"] for value in row):
                raise ValueError("execute operands cannot be strings. Encode a file path as {\"utf8\":\"input.bin\"}; numbers must be JSON integers, e.g. 53, not \"53\". Retry execute with corrected types.")
            if capability and any(len(row) != capability.plan.arity for row in args["operands"]):
                raise ValueError(f"{args['name']} requires {capability.plan.arity} operands per row; input types: {capability.provenance.get('input_types')}")
            operands = []
            for row in args["operands"]:
                converted = []
                for value in row:
                    if isinstance(value, dict):
                        if self.runtime.host is None:
                            raise ValueError("buffer operands require a host root")
                        if set(value) == {"utf8"}:
                            value = self.runtime.host.put(value["utf8"].encode("utf-8"))
                        elif set(value) == {"hex"}:
                            value = self.runtime.host.put(bytes.fromhex(value["hex"]))
                        else:
                            raise ValueError("invalid portable buffer operand")
                    converted.append(value)
                operands.append(tuple(converted))
            result = self.runtime.request(args["name"], operands, args["width"])
            if result.status == "executed":
                self.save()
            return asdict(result)
        raise ValueError(f"unknown agent tool {name}")

    def save(self):
        if self.catalog is not None:
            self.catalog.refresh()
        if self.program_path is not None:
            self.runtime.save(self.program_path)

    def run(self, intent: str, *, evidence: str | None = None):
        self._resolved, self._resolution_status = None, None
        self._proposed = None
        self._goal, self._candidates = None, []
        self._caller_evidence, self._caller_types = None, None
        self._verification_attempted = False
        if self.catalog is not None and evidence is not None:
            from vectorpro.semantic_catalog import operand_types
            try:
                supplied = json.loads(evidence)
            except (ValueError, TypeError):
                supplied = None  # free-form evidence can still be interpreted by the model
            if isinstance(supplied, dict) and set(supplied) == {"width", "operands", "targets"}:
                self._caller_types = operand_types(supplied["operands"])
                self._caller_evidence = {"numeric_validation": supplied}
            else:
                cases = supplied.get("cases") if isinstance(supplied, dict) else supplied
                if isinstance(cases, list) and cases and all(isinstance(c, dict) and "inputs" in c for c in cases):
                    self._caller_types = operand_types([c["inputs"] for c in cases])
                    self._caller_evidence = {"state_validation": cases}
        if self.catalog is None:
            self.messages.append({"role": "system", "content": "Available capabilities (names, arity and input types): " +
                                  json.dumps(self.call("list_capabilities", {}), ensure_ascii=False)})
        self.messages.append({"role": "user", "content": intent})
        events = []
        for _ in range(self.max_calls):
            offered_tools = self.available_tools()
            message = self.model.complete(self.messages, offered_tools)
            if message.get("role") != "assistant":
                raise ValueError("model must return an assistant message")
            self.messages.append(message)
            calls = message.get("tool_calls") or []
            if not calls:
                if self.catalog is not None and not any(e["result"].get("status") == "executed" for e in events):
                    status = self._resolution_status if self._resolution_status in ("needs_evidence", "needs_learning_examples") else "not_executed"
                    return {"status": status, "text": message.get("content"), "tools": events}
                return {"status": "answered", "text": message.get("content"), "tools": events}
            if len(calls) != 1:
                raise ValueError("agent accepts one sequential tool call per response")
            call = calls[0]
            try:
                if call["function"]["name"] not in {t["function"]["name"] for t in offered_tools}:
                    raise ValueError("tool is not available in the current request phase")
                args = json.loads(call["function"]["arguments"])
                result = self.call(call["function"]["name"], args)
            except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
                result = {"status": "error", "message": str(error)}
            events.append({"name": call["function"]["name"], "result": result})
            self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False)})
            if evidence is not None and (call["function"]["name"].startswith("prepare_") or call["function"]["name"] in ("propose_request", "propose_numeric_request", "propose_file_request")) and result.get("status") == "proposed":
                self.messages.append({"role": "user", "content": "Separate validation evidence from the caller (virtual only; DO NOT change the already proposed request):\n" + evidence})
                evidence = None
            if call["function"]["name"] == "ask_user" and result.get("status") == "needs_input":
                return result | {"tools": events}
        return {"status": "call_budget", "tools": events}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--host-root", type=Path)
    parser.add_argument("--endpoint", required=True, help="full chat/completions endpoint URL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="VECTORPRO_LLM_API_KEY")
    parser.add_argument("--intent", required=True)
    parser.add_argument("--encoder", type=Path, help="enable tensor-catalog resolution with a local multilingual MiniLM directory")
    parser.add_argument("--no-learning", action="store_true", help="execution-only session; do not expose teaching tools")
    parser.add_argument("--evidence-file", type=Path, help="caller evidence shown only after actual inputs have been proposed")
    args = parser.parse_args(argv)
    host = HostContext(args.host_root) if args.host_root else None
    runtime = VectorRuntime.load(args.program, host=host) if args.program.exists() else VectorRuntime(host=host)
    if host:
        runtime.provide_host_operations()
    model = HTTPChatModel(args.endpoint, args.model, os.environ.get(args.api_key_env))
    catalog = None
    if args.encoder:
        from vectorpro.semantic_catalog import Encoder, TensorCatalog
        identity = json.loads((args.encoder / "download.json").read_text())
        catalog = TensorCatalog(runtime, Encoder(args.encoder), identity)
    result = AgentSession(runtime, model, args.program, catalog=catalog, allow_learning=not args.no_learning).run(
        args.intent, evidence=args.evidence_file.read_text(encoding="utf-8") if args.evidence_file else None)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "answered" else 2


if __name__ == "__main__":
    raise SystemExit(main())
