"""Real local Gemma inference through HTTP and the unmodified agent tool loop.

Requires an explicitly installed llama-cpp-python and GGUF inside the ONE test
container. Native dialogue template with an explicit tool JSON protocol: no scripted replies,
no oracle-generated tool arguments and no replacement of the model's decisions.
"""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import random
import re
import shutil
from threading import Thread
from time import monotonic

from vectorpro.agent import AgentSession, HTTPChatModel
from vectorpro.host import HostContext
from vectorpro.runtime import VectorRuntime


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes().hex()
            for p in root.rglob("*") if p.is_file()}


def evaluate_case(case, response, runtime, native_root, initial_registry):
    expected = case["expect"]
    events = response.get("tools", [])
    executed = [e["result"] for e in events if e["result"].get("status") == "executed"]
    checks = {"files": snapshot(native_root) == expected["files"]}
    if expected["status"] == "executed":
        checks["output"] = any(e.get("outputs") == expected["outputs"] for e in executed)
        checks["terminal_status"] = response["status"] == "executed"
        if "request" in expected:
            actual = response.get("request") or {}
            checks["request_binding"] = all(actual.get(k) == v for k, v in expected["request"].items())
    else:
        checks["question"] = response["status"] == "needs_input"
        checks["no_native_effects"] = not executed and runtime.host.events == []
        checks["no_teaching"] = runtime.registry.to_data() == initial_registry and not any(e["name"].startswith("teach") for e in events)
        if expected["status"] == "unsupported":
            checks["verified_unsupported"] = any(e["result"].get("status") == "needs_learning_examples"
                and (e["result"].get("verification") or e["name"] in ("verify_numeric", "verify_state", "check_small"))
                for e in events)
    if "learned" in expected:
        name = expected["learned"]
        checks["learned"] = name in runtime.registry and any(e["name"].startswith("teach") and e["result"].get("status") == "learned" for e in events)
        proof = expected["heldout"]
        if checks["learned"]:
            result = runtime.request(name, [tuple(r) for r in proof["operands"]], proof["width"])
            checks["heldout"] = result.outputs == proof["targets"]
            restored = VectorRuntime.load(case["program"])
            checks["reload"] = restored.request(name, [tuple(r) for r in proof["operands"]], proof["width"]).outputs == proof["targets"]
        else:
            checks["heldout"] = checks["reload"] = False
    return {"passed": all(checks.values()), "checks": checks, "response": response}


def native_message(text):
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", cleaned, flags=re.S)
    if not matches:
        return {"role": "assistant", "content": cleaned}
    calls = []
    for i, encoded in enumerate(matches):
        data = json.loads(encoded)
        calls.append({"id": f"local_call_{i}", "type": "function", "function": {
            "name": data["name"], "arguments": json.dumps(data["arguments"], ensure_ascii=False)}})
    return {"role": "assistant", "content": None, "tool_calls": calls}


def gemma_messages(messages, tools):
    """Adapt role/tool framing only; preserve instructions, evidence and arguments."""
    guide = ('Available tools (JSON schemas):\n' + json.dumps(tools, ensure_ascii=False)
             + '\nTo call ONE tool output <tool_call>{"name":"tool_name","arguments":{...}}</tool_call>.'
             + '\nChoose the tool and all argument values yourself. Tool results are user messages labeled TOOL_RESULT.'
             + '\nAfter an executed result, report the actual result in plain language.')
    result = []
    for message in messages:
        role = message["role"]
        content = message.get("content") or ""
        if role == "system":
            role, content = "user", "INSTRUCTIONS:\n" + content
        elif role == "tool":
            role, content = "user", "TOOL_RESULT " + message["tool_call_id"] + ":\n" + content
        elif message.get("tool_calls"):
            content += "".join('<tool_call>' + json.dumps({"name": call["function"]["name"],
                "arguments": json.loads(call["function"]["arguments"])}, ensure_ascii=False) + '</tool_call>'
                for call in message["tool_calls"])
        if result and result[-1]["role"] == role:
            result[-1]["content"] += "\n\n" + content
        else:
            result.append({"role": role, "content": content})
    if not result or result[0]["role"] != "user":
        raise ValueError("Gemma dialogue must start with user instructions")
    result[0]["content"] = guide + "\n\n" + result[0]["content"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/opt/vectorpro-models/gemma-3-1b-it-Q8_0.gguf")
    parser.add_argument("--root", type=Path, default=Path("results/local_small_llm"))
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--catalog", type=Path, help="use a tensor program and the evidence-gated catalog tool loop")
    parser.add_argument("--encoder", default="/opt/vectorpro-models/multilingual-minilm")
    parser.add_argument("--constrain-tools", action="store_true", help="Use JSON-schema constrained decoding for tool turns; model still chooses tool and arguments")
    parser.add_argument("--max-tokens", type=int, default=512, help="bound generated output per model turn")
    parser.add_argument("--evaluation", type=Path, help="data-only cases; expected outputs are never passed to the model")
    parser.add_argument("--evaluation-kind", choices=("replay", "first-use"), default="replay",
                        help="record a genuinely new frozen evaluation explicitly; repeated requests remain replay")
    parser.add_argument("--max-calls", type=int, default=8)
    args = parser.parse_args()
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8")) if args.evaluation else None
    cases = {c["name"]: c for c in evaluation["cases"]} if evaluation else {}
    if evaluation:
        if not args.catalog or len(cases) != len(evaluation["cases"]):
            parser.error("evaluation needs a catalog and unique case names")
        if args.root.exists():
            parser.error("evaluation output root already exists; preserve it and choose a new root")
    if args.scenarios is None:
        args.scenarios = list(cases) if evaluation else (["catalog_xor", "catalog_sub", "catalog_file", "catalog_unknown",
                           "catalog_delete", "catalog_clarify", "catalog_xor_self", "catalog_korean"]
                          if args.catalog else ["known", "clarify", "file", "learn"])
    encoder, identity = None, None
    if args.catalog:
        from vectorpro.semantic_catalog import Encoder, TensorCatalog
        encoder = Encoder(args.encoder)
        identity = json.loads((Path(args.encoder) / "download.json").read_text())
    import llama_cpp
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter
    llm = llama_cpp.Llama(model_path=args.model, n_ctx=8192, n_threads=4, n_batch=256, verbose=False, seed=0)
    if not llm.metadata.get("general.architecture", "").startswith("gemma"):
        raise ValueError("this local adapter expects a Gemma GGUF")
    formatter = Jinja2ChatFormatter(llm.metadata["tokenizer.chat_template"],
                                   eos_token="<eos>", bos_token="<bos>")
    raw_calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                template_messages = gemma_messages(data["messages"], data["tools"])
                formatted = formatter(messages=template_messages)
                started = monotonic()
                grammar = None
                last = data["messages"][-1]
                completed_execution = (last.get("role") == "tool" and json.loads(last["content"]).get("status") == "executed")
                if args.constrain_tools and not completed_execution:
                    schema = {"anyOf": [{"type": "object", "properties": {
                        "name": {"const": t["function"]["name"]},
                        "arguments": t["function"]["parameters"]},
                        "required": ["name", "arguments"], "additionalProperties": False} for t in data["tools"]]}
                    json_grammar = llama_cpp.LlamaGrammar.from_json_schema(json.dumps(schema), verbose=False)._grammar
                    json_grammar = re.sub(r"(?m)^root ::=", "tool-json ::=", json_grammar)
                    grammar = llama_cpp.LlamaGrammar.from_string('root ::= "<tool_call>" tool-json "</tool_call>"\n' + json_grammar, verbose=False)
                # The native template already includes <bos>; avoid adding a second BOS.
                prompt_tokens = llm.tokenize(formatted.prompt.encode("utf-8"), add_bos=False, special=True)
                generated = llm.create_completion(prompt_tokens, max_tokens=args.max_tokens, temperature=0,
                                                  stop=["<end_of_turn>", "<eos>"], seed=0, grammar=grammar)
                text = generated["choices"][0]["text"]
                raw_calls.append({"messages": data["messages"], "raw_output": text,
                                  "usage": generated.get("usage"), "seconds": monotonic() - started})
                message = native_message(text)
                payload = json.dumps({"choices": [{"message": message}]}).encode()
                self.send_response(200)
            except Exception as error:
                raw_calls.append({"server_error": repr(error)})
                payload = json.dumps({"error": str(error)}).encode()
                self.send_response(500)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    args.root.mkdir(parents=True, exist_ok=True)
    prompts = {
        "known": "Use the acquired xor function to compute bitwise XOR of 12345 and 4567 at width 16. Do not teach a function. Use tools, then report the actual result.",
        "clarify": "Do something with input.bin. I have not decided what transformation or output I want. Ask me what is missing instead of modifying the file.",
        "file": "Use the acquired map function on input.bin with mask 53 at width 16. Do not teach anything. Change only this file and report the tool result.",
        "learn": "Acquire a NEW function named tiny_nand: W-bit bitwise complement of AND of two inputs. Make your own numeric training and validation examples. Use a 4-bit training set and a distinct 8-bit validation set. Teach it using the documented tool, then execute tiny_nand on inputs 5 and 3 at width 8. Use the actual tools and do not claim learning without a successful teach result.",
        "catalog_xor": "Compute bitwise XOR of 12345 and 4567 at width 16. Correct validation examples at width 4: [1,3] gives 2 and [5,3] gives 6. Find a verified stored function, then execute the requested inputs. Do not teach.",
        "catalog_sub": "Subtract 4567 from 12345 at width 16. Correct validation examples at width 4: [5,3] gives 2 and [9,4] gives 5. Resolve the request then execute it; do not teach.",
        "catalog_file": 'ACTUAL REQUEST: XOR each byte of input.bin with mask 53 at width 16 and save to the same file.\nVIRTUAL VALIDATION ONLY: {"inputs":[{"utf8":"probe.bin"},1],"before":{"probe.bin":"0007"},"after":{"probe.bin":"0106"},"output":2}.\nUse the actual request paths/values in step 1 and this validation in step 2. Do not teach.',
        "catalog_unknown": "Multiply 5 and 3 at width 16. Correct validation examples at width 4: [1,3] gives 3 and [5,3] gives 15. Check whether a stored function actually matches. Do not teach or substitute another operation. If none matches, ask me for learning evidence.",
        "catalog_delete": "Delete input.bin. At width 16 a correct validation demonstration is: input probe.bin, before contains probe.bin hex 0007, after is the empty file snapshot. Resolve this request. Do not teach or substitute overwriting for deletion. Ask me if no stored function matches.",
        "catalog_clarify": "Do something with input.bin. I have not chosen the transformation or desired output. Ask me what I want; leave files unchanged.",
        "catalog_xor_self": "Compute bitwise exclusive OR of 12345 and 4567 at width 16. Generate two correct small numeric validation examples yourself to resolve a stored function, then execute it. Do not teach.",
        "catalog_korean": "12345와 4567의 비트별 배타적 논리합을 16비트로 계산해줘. 검증 예제는 4비트에서 [1,3]의 정답 2, [5,3]의 정답 6이야. 학습하지 말고 저장된 기능을 검증한 다음 실제 입력을 실행해줘."}
    catalog_intents = {
        "catalog_xor": "Compute bitwise XOR of 12345 and 4567 at width 16. Resolve a stored function then execute it. Do not teach.",
        "catalog_sub": "Subtract 4567 from 12345 at width 16. Resolve then execute it. Do not teach.",
        "catalog_file": "XOR every byte of input.bin with mask 53 at width 16 and save to the same file. Resolve then execute. Do not teach.",
        "catalog_unknown": "Multiply 5 and 3 at width 16. Check for a stored function. Do not teach or substitute another operation. Ask me if unsupported.",
        "catalog_delete": "Delete input.bin at width 16. Do not teach or substitute another operation. Ask me if unsupported.",
        "catalog_korean": "12345와 4567의 비트별 배타적 논리합을 16비트로 계산해줘. 저장된 기능을 검증하고 실행해줘. 학습하지 마."}
    supplied_evidence = {
        "catalog_xor": {"width": 4, "operands": [[1,3],[5,3]], "targets": [2,6]},
        "catalog_korean": {"width": 4, "operands": [[1,3],[5,3]], "targets": [2,6]},
        "catalog_sub": {"width": 4, "operands": [[5,3],[9,4]], "targets": [2,5]},
        "catalog_unknown": {"width": 4, "operands": [[1,3],[5,3]], "targets": [3,15]},
        "catalog_file": {"cases": [{"inputs":[{"utf8":"probe.bin"},1],"before":{"probe.bin":"0007"},"after":{"probe.bin":"0106"},"output":2}]},
        "catalog_delete": {"cases": [{"inputs":[{"utf8":"probe.bin"}],"before":{"probe.bin":"0007"},"after":{}}]}}
    results = {}
    try:
        for name in args.scenarios:
            root = args.root / name
            root.mkdir(parents=True, exist_ok=True)
            program = root / ("program.pt" if args.catalog else "program.json")
            shutil.copyfile(args.catalog or "results/initial_model/program.json", program)
            case = cases.get(name)
            native_root = root / "native" if case else root
            native_root.mkdir(exist_ok=True)
            files = case["before"] if case else {"input.bin": "0007ff80"}
            from vectorpro.host import MemoryHostContext
            for path, hex_bytes in files.items():
                target = native_root / MemoryHostContext.normalize(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(bytes.fromhex(hex_bytes))
            model = HTTPChatModel(f"http://127.0.0.1:{server.server_port}/v1/chat/completions", Path(args.model).stem, timeout=180)
            runtime = VectorRuntime.load(program, host=HostContext(native_root))
            catalog = TensorCatalog(runtime, encoder, identity) if args.catalog else None
            session = AgentSession(runtime, model, program, max_calls=args.max_calls, catalog=catalog,
                                   allow_learning=case.get("allow_learning", False) if case else not bool(args.catalog))
            initial_registry = runtime.registry.to_data()
            start = len(raw_calls)
            print(f"running {name}", flush=True)
            try:
                intent = case["intent"] if case else catalog_intents.get(name, prompts[name]) if args.catalog else prompts[name]
                evidence = case.get("evidence") if case else supplied_evidence.get(name) if args.catalog else None
                response = session.run(intent, evidence=json.dumps(evidence) if evidence is not None else None)
                if case:
                    results[name] = evaluate_case({**case, "program": program}, response, runtime, native_root, initial_registry)
                    (root / "transcript.json").write_text(json.dumps(raw_calls[start:], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    (root / "result.json").write_text(json.dumps(results[name], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    print(json.dumps({"scenario": name, "passed": results[name]["passed"]}), flush=True)
                    continue
                tools = response.get("tools", [])
                outputs = [e["result"].get("outputs") for e in tools if isinstance(e["result"], dict)]
                if name in ("known", "catalog_xor", "catalog_xor_self", "catalog_korean"):
                    passed = [12345 ^ 4567] in outputs and not any(e["name"] == "teach" for e in tools)
                elif name in ("clarify", "catalog_clarify"):
                    passed = response["status"] == "needs_input" and (root / "input.bin").read_bytes() == b"\x00\x07\xff\x80"
                elif name in ("file", "catalog_file"):
                    passed = [4] in outputs and (root / "input.bin").read_bytes() == b"\x35\x32\xca\xb5"
                elif name == "catalog_sub":
                    passed = [12345 - 4567] in outputs
                elif name in ("catalog_unknown", "catalog_delete"):
                    resolutions = [e["result"] for e in tools if e["name"] in ("search_goal", "resolve_request", "resolve_numeric", "resolve_state", "verify_numeric", "verify_state", "check_small")]
                    passed = (response["status"] == "needs_input" and any(r.get("status") == "needs_learning_examples" for r in resolutions)
                              and not any(e["name"] in ("execute", "execute_resolved", "teach", "teach_numeric") for e in tools)
                              and (root / "input.bin").read_bytes() == b"\x00\x07\xff\x80")
                else:
                    passed = [254] in outputs and "tiny_nand" in runtime.registry
                    held_out = 0
                    if "tiny_nand" in runtime.registry:
                        restored = VectorRuntime.load(program)
                        rng = random.Random(21)
                        for _ in range(100):
                            a, b = rng.getrandbits(16), rng.getrandbits(16)
                            held_out += restored.request("tiny_nand", [(a, b)], 16).outputs == [(~(a & b)) & 65535]
                    passed = passed and held_out == 100
                    response["held_out"] = held_out
                results[name] = {"passed": bool(passed), "response": response}
            except Exception as error:
                results[name] = {"passed": False, "error": str(error)}
            (root / "transcript.json").write_text(json.dumps(raw_calls[start:], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (root / "result.json").write_text(json.dumps(results[name], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"scenario": name, "passed": results[name]["passed"]}), flush=True)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    with Path(args.model).open("rb") as stream:
        model_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest_path = Path(args.model).parent / "gemma-download.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest.get("sha256") != model_digest:
        manifest = {}  # never label an arbitrary --model file as the pinned download
    summary = {"model": manifest.get("repo", Path(args.model).name), "revision": manifest.get("revision"),
               "quantization": manifest.get("quantization"), "model_bytes": Path(args.model).stat().st_size,
               "model_sha256": model_digest, "chat_protocol": "Gemma native template with explicit JSON tool schemas",
               "llama_cpp_python": llama_cpp.__version__, "real_model_inference": True,
               "scripted_replies": False, "container": "vectorpro-test", "results": results}
    summary["constrained_tool_decoding"] = args.constrain_tools
    summary["tensor_catalog"] = str(args.catalog) if args.catalog else None
    summary["caller_evidence_delivered_after_input_extraction"] = bool(args.catalog)
    summary["max_tokens_per_turn"] = args.max_tokens
    summary["evaluation_protocol"] = "development scenarios reused during adapter improvements; not independent held-out accuracy"
    if evaluation:
        summary["dataset_protocol"] = evaluation["protocol"]
        summary["evaluation_protocol"] = ("first-use frozen requests after development; no retuning on results"
            if args.evaluation_kind == "first-use" else "replay of existing fixed requests; not new independent held-out accuracy")
        summary["evaluation_kind"] = args.evaluation_kind
        summary["evaluation_file"] = str(args.evaluation)
        summary["evaluation_sha256"] = hashlib.sha256(args.evaluation.read_bytes()).hexdigest()
    summary["max_calls"] = args.max_calls
    (args.root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": sum(r["passed"] for r in results.values()), "total": len(results)}), flush=True)


if __name__ == "__main__":
    main()
