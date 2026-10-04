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
                 max_calls: int = 12):
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError("max_calls must be a positive integer")
        self.runtime, self.model, self.program_path, self.max_calls = runtime, model, program_path, max_calls
        self.messages = [{"role": "system", "content": GUIDE}]

    def call(self, name, args):
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
                self.runtime.registry.get(data["name"]).provenance["evidence_source"] = "LLM adapter supplied examples"
                self.save()
            return asdict(result)
        if name == "execute":
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
        if self.program_path is not None:
            self.runtime.save(self.program_path)

    def run(self, intent: str):
        self.messages.append({"role": "system", "content": "Available capabilities (names, arity and input types): " +
                              json.dumps(self.call("list_capabilities", {}), ensure_ascii=False)})
        self.messages.append({"role": "user", "content": intent})
        events = []
        for _ in range(self.max_calls):
            message = self.model.complete(self.messages, TOOLS)
            if message.get("role") != "assistant":
                raise ValueError("model must return an assistant message")
            self.messages.append(message)
            calls = message.get("tool_calls") or []
            if not calls:
                return {"status": "answered", "text": message.get("content"), "tools": events}
            if len(calls) != 1:
                raise ValueError("agent accepts one sequential tool call per response")
            call = calls[0]
            try:
                args = json.loads(call["function"]["arguments"])
                result = self.call(call["function"]["name"], args)
            except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
                result = {"status": "error", "message": str(error)}
            events.append({"name": call["function"]["name"], "result": result})
            self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False)})
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
    args = parser.parse_args(argv)
    host = HostContext(args.host_root) if args.host_root else None
    runtime = VectorRuntime.load(args.program, host=host) if args.program.exists() else VectorRuntime(host=host)
    if host:
        runtime.provide_host_operations()
    model = HTTPChatModel(args.endpoint, args.model, os.environ.get(args.api_key_env))
    result = AgentSession(runtime, model, args.program).run(args.intent)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "answered" else 2


if __name__ == "__main__":
    raise SystemExit(main())
