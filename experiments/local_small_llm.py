"""Real local Qwen inference through HTTP and the unmodified agent tool loop.

Requires an explicitly installed llama-cpp-python and GGUF inside the ONE test
container. Native model template/tool-call parsing only: no scripted replies,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/opt/vectorpro-models/Qwen3-0.6B-Q8_0.gguf")
    parser.add_argument("--root", type=Path, default=Path("results/local_small_llm"))
    parser.add_argument("--scenarios", nargs="+", default=["known", "clarify", "file", "learn"])
    parser.add_argument("--constrain-tools", action="store_true", help="Use JSON-schema constrained decoding for tool turns; model still chooses tool and arguments")
    args = parser.parse_args()
    import llama_cpp
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter
    llm = llama_cpp.Llama(model_path=args.model, n_ctx=8192, n_threads=4, n_batch=256, verbose=False, seed=0)
    formatter = Jinja2ChatFormatter(llm.metadata["tokenizer.chat_template"],
                                   eos_token="<|im_end|>", bos_token="<|endoftext|>")
    raw_calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                # The native Qwen template expects string content even for tool calls.
                template_messages = [dict(m, content=m.get("content") or "") for m in data["messages"]]
                formatted = formatter(messages=template_messages, tools=data["tools"], enable_thinking=False)
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
                generated = llm.create_completion(formatted.prompt, max_tokens=1200, temperature=0,
                                                  stop=["<|im_end|>"], seed=0, grammar=grammar)
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
        "learn": "Acquire a NEW function named tiny_nand: W-bit bitwise complement of AND of two inputs. Make your own numeric training and validation examples. Use a 4-bit training set and a distinct 8-bit validation set. Teach it using the documented tool, then execute tiny_nand on inputs 5 and 3 at width 8. Use the actual tools and do not claim learning without a successful teach result."}
    results = {}
    try:
        for name in args.scenarios:
            root = args.root / name
            root.mkdir(parents=True, exist_ok=True)
            program = root / "program.json"
            shutil.copyfile("results/initial_model/program.json", program)
            (root / "input.bin").write_bytes(b"\x00\x07\xff\x80")
            model = HTTPChatModel(f"http://127.0.0.1:{server.server_port}/v1/chat/completions", "Qwen3-0.6B-Q8_0", timeout=180)
            runtime = VectorRuntime.load(program, host=HostContext(root))
            session = AgentSession(runtime, model, program, max_calls=8)
            start = len(raw_calls)
            print(f"running {name}", flush=True)
            try:
                response = session.run(prompts[name])
                tools = response.get("tools", [])
                outputs = [e["result"].get("outputs") for e in tools if isinstance(e["result"], dict)]
                if name == "known":
                    passed = [12345 ^ 4567] in outputs and not any(e["name"] == "teach" for e in tools)
                elif name == "clarify":
                    passed = response["status"] == "needs_input" and (root / "input.bin").read_bytes() == b"\x00\x07\xff\x80"
                elif name == "file":
                    passed = [4] in outputs and (root / "input.bin").read_bytes() == b"\x35\x32\xca\xb5"
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
    summary = {"model": "Qwen/Qwen3-0.6B-GGUF", "revision": "23749fefcc72300e3a2ad315e1317431b06b590a",
               "quantization": "Q8_0", "model_bytes": Path(args.model).stat().st_size,
               "model_sha256": hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
               "llama_cpp_python": llama_cpp.__version__, "real_model_inference": True,
               "scripted_replies": False, "container": "vectorpro-test", "results": results}
    summary["constrained_tool_decoding"] = args.constrain_tools
    (args.root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": sum(r["passed"] for r in results.values()), "total": len(results)}), flush=True)


if __name__ == "__main__":
    main()
