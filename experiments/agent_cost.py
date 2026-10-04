"""Measured Gemma orchestration versus learned contracts and a Python macro control."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
from time import perf_counter

from local_small_llm import gemma_messages, native_message
from linux_filesystem import curricula, snapshot
from vectorpro.agent import tool
from vectorpro.host import HostContext
from vectorpro.runtime import VectorRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/agent_cost"))
    parser.add_argument("--model", default="/opt/vectorpro-models/gemma-3-1b-it-Q8_0.gguf")
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("preserve existing results; choose a new output path")
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    args.output.mkdir(parents=True)
    cases = [{"name": f"case-{i}", "folder": f"job-{i}", "source": f"source-{i}.bin",
              "destination": f"job-{i}/result.bin", "data": bytes(range(n)) if n <= 256 else b"x" * n}
             for i, n in enumerate([0, 7, 32, 128, 256, 4096])]
    request = next(curricula())
    protocol = {"task": "create a new folder and copy an existing file into it",
                "cases": [{k: v.hex() if k == "data" else v for k, v in c.items()} for c in cases],
                "training": request, "model": args.model, "repeats": args.repeats,
                "max_llm_calls": 6, "temperature": 0, "seed": 0,
                "interpretation": "token and latency proxies, no billed dollars/energy; supplied examples exclude evidence authoring cost"}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    started = perf_counter()
    runtime = VectorRuntime()
    learned = runtime.teach_contract(**request)
    assert learned["status"] == "registered", learned
    learning_seconds = perf_counter() - started
    program = args.output / "program.pt"
    started = perf_counter()
    runtime.save(program)
    save_seconds = perf_counter() - started
    started = perf_counter()
    VectorRuntime.load(program)
    load_seconds = perf_counter() - started
    identity = learned["contract"]["id"]

    def setup(case, root):
        root.mkdir(parents=True)
        (root / case["source"]).write_bytes(case["data"])
        (root / "keep").write_bytes(b"preserved")

    def check(case, root):
        return snapshot(root) == {"files": {case["source"]: case["data"].hex(), "keep": b"preserved".hex(),
                                           case["destination"]: case["data"].hex()}, "directories": [case["folder"]]}

    def macro(root, arguments):
        (root / arguments["folder"]).mkdir()
        data = (root / arguments["source"]).read_bytes()
        return (root / arguments["destination"]).write_bytes(data)

    direct = []
    for case in cases:
        arguments = {k: case[k] for k in ("folder", "source", "destination")}
        for repeat in range(args.repeats):
            for mode in (["direct_tensor", "direct_python"] if repeat % 2 == 0 else ["direct_python", "direct_tensor"]):
                root = args.output / "native" / f"{case['name']}-{mode}-{repeat}"
                setup(case, root)
                runtime.host = HostContext(root)
                started = perf_counter()
                output = (runtime.call_contract(identity, arguments, 16)["outputs"][0]
                          if mode == "direct_tensor" else macro(root, arguments))
                seconds = perf_counter() - started
                direct.append({"case": case["name"], "mode": mode, "seconds": seconds,
                               "passed": output == len(case["data"]) and check(case, root),
                               "llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
    (args.output / "direct.json").write_text(json.dumps(direct, indent=2))
    import llama_cpp
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter
    started = perf_counter()
    model = llama_cpp.Llama(model_path=args.model, n_ctx=8192, n_threads=4, n_batch=256, verbose=False, seed=0)
    model_load_seconds = perf_counter() - started
    formatter = Jinja2ChatFormatter(model.metadata["tokenizer.chat_template"], eos_token="<eos>", bos_token="<bos>")
    string = {"type": "string"}
    primitives = [tool("create_directory", "Create one new directory", {"path": string}, ("path",)),
                  tool("read_file", "Read a file and return a transient buffer handle", {"path": string}, ("path",)),
                  tool("write_file", "Write a previously read buffer to a file", {"path": string, "buffer": {"type": "integer"}}, ("path", "buffer"))]
    combined = [tool("make_copy", "Create folder, read source and write bytes to destination", {k: string for k in ("folder", "source", "destination")}, ("folder", "source", "destination"))]
    traces = []
    def infer(messages, tools):
        formatted = formatter(messages=gemma_messages(messages, tools))
        schema = {"anyOf": [{"type": "object", "properties": {"name": {"const": t["function"]["name"]},
                  "arguments": t["function"]["parameters"]}, "required": ["name", "arguments"], "additionalProperties": False} for t in tools]}
        grammar_text = llama_cpp.LlamaGrammar.from_json_schema(json.dumps(schema), verbose=False)._grammar
        grammar_text = re.sub(r"(?m)^root ::=", "tool-json ::=", grammar_text)
        grammar = llama_cpp.LlamaGrammar.from_string('root ::= "<tool_call>" tool-json "</tool_call>"\n' + grammar_text, verbose=False)
        started = perf_counter()
        generated = model.create_completion(model.tokenize(formatted.prompt.encode(), add_bos=False, special=True),
                    max_tokens=256, temperature=0, seed=0, grammar=grammar, stop=["<end_of_turn>", "<eos>"])
        raw = generated["choices"][0]["text"]
        try:
            message = native_message(raw)
        except (ValueError, KeyError, TypeError) as error:
            message = {"role": "assistant", "content": raw, "parse_error": str(error)}
        return message, generated["usage"], perf_counter() - started, raw
    # Exclude cold initialization/warmup from all repeat latency; report it separately.
    started = perf_counter()
    model.create_completion("Hello", max_tokens=1, temperature=0)
    warmup_seconds = perf_counter() - started
    jobs = [(case, mode) for case in cases for mode in ("llm_primitives", "llm_tensor", "llm_python_macro")]
    random.Random(20261004).shuffle(jobs)
    for case, mode in jobs:
        root = args.output / "native" / f"{case['name']}-{mode}"
        setup(case, root)
        runtime.host = context = HostContext(root)
        tools = primitives if mode == "llm_primitives" else combined
        messages = [{"role": "system", "content": "Use one tool at a time to perform the actual requested file operation. Use only provided paths. After an error correct your arguments. Never invent file contents."},
                    {"role": "user", "content": f"Create directory {case['folder']} and copy all bytes of {case['source']} into {case['destination']}. Preserve all other files."}]
        event = {"case": case["name"], "mode": mode, "calls": [], "passed": False}
        started = perf_counter()
        for step in range(6):
            completed = False
            try:
                message, usage, seconds, raw = infer(messages, tools)
                entry = {"usage": usage, "seconds": seconds, "raw": raw}
                event["calls"].append(entry)
                messages.append(message)
                call = message["tool_calls"][0]
                name = call["function"]["name"]
                values = json.loads(call["function"]["arguments"])
                expected_keys = {"path", "buffer"} if name == "write_file" else {"folder", "source", "destination"} if name == "make_copy" else {"path"}
                if set(values) != expected_keys:
                    raise ValueError("wrong tool arguments")
                # Validate every path before any operation, for both baseline and tensor.
                for key, value in values.items():
                    if key != "buffer":
                        context._path(context.put(value.encode()))
                if name == "make_copy" and mode != "llm_primitives":
                    output = (runtime.call_contract(identity, values, 16)["outputs"][0] if mode == "llm_tensor" else macro(root, values))
                    result = {"output": output}
                    completed = True
                elif name in ("create_directory", "read_file", "write_file") and mode == "llm_primitives":
                    operation = {"create_directory": "directory.create", "read_file": "file.read", "write_file": "file.write"}[name]
                    inputs = (context.put(values["path"].encode()),)
                    if name == "write_file":
                        inputs += (values["buffer"],)
                    output = context.invoke(operation, inputs, 16)
                    result = {"buffer": output} if name == "read_file" else {"output": output}
                    completed = name == "write_file"
                else:
                    raise ValueError("tool not offered")
                entry["result"] = result
            except Exception as error:
                result = {"error": str(error)}
                if event["calls"]:
                    event["calls"][-1]["error"] = str(error)
            if completed:
                event["passed"] = check(case, root)
                break
            messages.append({"role": "user", "content": "TOOL_RESULT: " + json.dumps(result)})
        event["seconds"] = perf_counter() - started
        event["messages"] = messages
        event["final_snapshot"] = snapshot(root)
        traces.append(event)
        (args.output / "llm.json").write_text(json.dumps(traces, indent=2), encoding="utf-8")
        print(json.dumps({k: event[k] for k in ("case", "mode", "seconds", "passed")}), flush=True)
    aggregates = {}
    for mode in ("llm_primitives", "llm_tensor", "llm_python_macro", "direct_tensor", "direct_python"):
        rows = [r for r in traces + direct if r["mode"] == mode]
        aggregates[mode] = {"attempts": len(rows), "passed": sum(r["passed"] for r in rows),
            "seconds_total": sum(r["seconds"] for r in rows), "seconds_median": statistics.median(r["seconds"] for r in rows),
            "llm_calls": sum(len(r.get("calls", [])) for r in rows),
            "prompt_tokens": sum(c["usage"]["prompt_tokens"] for r in rows for c in r.get("calls", [])),
            "completion_tokens": sum(c["usage"]["completion_tokens"] for r in rows for c in r.get("calls", []))}
    report = {"model": args.model, "model_load_seconds": model_load_seconds, "warmup_seconds": warmup_seconds,
              "learning_seconds": learning_seconds, "save_seconds": save_seconds, "program_load_seconds": load_seconds,
              "program_sha256": hashlib.sha256(program.read_bytes()).hexdigest(), "aggregates": aggregates,
              "limitations": ["6 fixed simple structured workflows, not arbitrary agent tasks", "No billed monetary or power cost", "Evidence authorship excluded", "Direct calls receive structured inputs; natural-language interpretation is bypassed", "Python macro is hand-written control, not a learned program", "Model prompt caching may benefit repeats; randomized mode order, warmup excluded"]}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
