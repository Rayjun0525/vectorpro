"""First bounded live-Gemma evaluation of caller-backed acquisition tools."""
import argparse
import json
import re
from pathlib import Path
from time import monotonic
from vectorpro.acquisition import EvidenceBank
from vectorpro.agent import AgentSession
from vectorpro.host import HostContext
from vectorpro.runtime import VectorRuntime
from experiments.local_small_llm import gemma_messages, native_message
from experiments.verified_acquisition import record, INTENT


class GemmaModel:
    def __init__(self, model_path):
        import llama_cpp
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter
        self.llama_cpp = llama_cpp
        self.llm = llama_cpp.Llama(model_path=model_path, n_ctx=8192, n_threads=4, n_batch=256,
                                 verbose=False, seed=0)
        if not self.llm.metadata.get("general.architecture", "").startswith("gemma"):
            raise ValueError("expected existing Gemma GGUF")
        self.formatter = Jinja2ChatFormatter(self.llm.metadata["tokenizer.chat_template"],
                                            eos_token="<eos>", bos_token="<bos>")
        self.raw = []

    def complete(self, messages, tools):
        schema = {"anyOf": [{"type": "object", "properties": {
            "name": {"const": t["function"]["name"]}, "arguments": t["function"]["parameters"]},
            "required": ["name", "arguments"], "additionalProperties": False} for t in tools]}
        grammar = self.llama_cpp.LlamaGrammar.from_json_schema(json.dumps(schema), verbose=False)._grammar
        grammar = re.sub(r"(?m)^root ::=", "tool-json ::=", grammar)
        constrained = self.llama_cpp.LlamaGrammar.from_string(
            'root ::= "<tool_call>" tool-json "</tool_call>"\n' + grammar, verbose=False)
        formatted = self.formatter(messages=gemma_messages(messages, tools))
        tokens = self.llm.tokenize(formatted.prompt.encode(), add_bos=False, special=True)
        start = monotonic()
        output = self.llm.create_completion(tokens, max_tokens=600, temperature=0, seed=0,
            stop=["<end_of_turn>", "<eos>"], grammar=constrained)
        text = output["choices"][0]["text"]
        self.raw.append({"messages": messages, "tools": tools, "raw_output": text,
                         "seconds": monotonic() - start, "usage": output.get("usage")})
        return native_message(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/verified_acquisition_gemma"))
    parser.add_argument("--model", default="/opt/vectorpro-models/gemma-3-1b-it-Q8_0.gguf")
    parser.add_argument("--run-kind", choices=("first-use", "replay"), default="first-use")
    args = parser.parse_args()
    root = args.output
    if root.exists():
        raise RuntimeError("Preserve evidence; choose a new output directory")
    root.mkdir(parents=True)
    source = record()
    (root / "evidence.json").write_text(json.dumps([source], indent=2), encoding="utf-8")
    # Fix inputs and expected states before any inference.
    cases = [("unknown", INTENT, b"first unseen\0\xff"),
             ("known", INTENT, "재사용 자료".encode()),
             ("missing_evidence", "input.bin의 각 바이트를 뒤집어서 output.bin에 저장해줘.", b"unchanged")]
    (root / "cases.json").write_text(json.dumps([
        {"name": n, "intent": intent, "input": data.hex()} for n, intent, data in cases], indent=2), encoding="utf-8")
    model = GemmaModel(args.model)
    outcomes = []
    program = root / "program.pt"
    runtime = VectorRuntime()
    for name, intent, data in cases:
        native = root / name
        native.mkdir()
        (native / "input.bin").write_bytes(data)
        (native / "keep").write_bytes(b"preserved")
        (native / "empty").mkdir()
        runtime.host = HostContext(native)
        before_registry = runtime.registry.to_data()
        try:
            result = AgentSession(runtime, model, program, max_calls=6,
                evidence_bank=EvidenceBank([source]), allow_learning=name != "known").run(intent)
        except (ValueError, RuntimeError, KeyError) as error:
            result = {"status": "error", "message": str(error)}
        files = {p.name: p.read_bytes().hex() for p in native.iterdir() if p.is_file()}
        directories = [p.name for p in native.iterdir() if p.is_dir()]
        expected = {"input.bin": data.hex(), "keep": b"preserved".hex()}
        if name == "missing_evidence":
            passed = result["status"] in ("needs_input", "not_executed") and files == expected and runtime.registry.to_data() == before_registry
        else:
            expected["output.bin"] = data.hex()
            passed = result["status"] == "executed" and result["outputs"] == [len(data)] and files == expected
            if name == "known":
                passed = passed and all(t["name"] != "learn_verified_contract" for t in result["tools"])
        outcomes.append({"name": name, "passed": passed and directories == ["empty"], "result": result,
                         "files": files, "directories": directories})
        (root / "raw_calls.json").write_text(json.dumps(model.raw, indent=2), encoding="utf-8")
        (root / "summary.json").write_text(json.dumps({"protocol": "bounded live Gemma; caller reference evidence", "run_kind": args.run_kind,
            "passed": sum(o["passed"] for o in outcomes), "cases": outcomes}, indent=2), encoding="utf-8")
        print(json.dumps({"case": name, "passed": outcomes[-1]["passed"], "status": result["status"]}), flush=True)


if __name__ == "__main__":
    main()
