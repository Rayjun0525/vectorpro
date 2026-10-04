"""Real Laya decision/binding replay; effects run ONLY in a memory filesystem.

This is not a ChatModel, contract generator, or autonomous teaching adapter.
Expected outputs and caller evidence are never fed to the decision model.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from time import monotonic

from vectorpro.agent import path_literals
from vectorpro.host import MemoryHostContext
from vectorpro.runtime import VectorRuntime
from vectorpro.semantic_catalog import argument_roles, convert, document, input_types


def choice(instructions, values):
    return {"type": "choice", "instructions": instructions,
            "criteria": {str(i): str(v) for i, v in enumerate(values)}}


def literals(intent):
    paths = path_literals(intent)
    text = list(intent)
    for start, end, _ in paths:
        text[start:end] = " " * (end - start)
    numbers = list(dict.fromkeys(int(s) for s in re.findall(r"(?<!\d)[+-]?\d+(?!\d)", "".join(text))))
    return list(dict.fromkeys(p[2] for p in paths)), numbers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/opt/vectorpro-models/laya-multilingual")
    parser.add_argument("--program", default="results/catalog_retrieval/final/program.pt")
    parser.add_argument("--evaluation", nargs="+", default=["experiments/requests/gemma_paths_first_use.json", "experiments/requests/gemma_binding_first_use.json"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve previous results; choose a new output file")
    import laya
    import torch
    torch.set_num_threads(4)
    manifest = json.loads((Path(args.model) / "download.json").read_text())
    model = laya.load(args.model, device="cpu", fast=False, compile=False,
                      expected_sha256={e["path"]: e["sha256"] for e in manifest["files"]})
    runtime = VectorRuntime.load(args.program)
    capabilities = list(runtime.registry)
    descriptions = [document(runtime.registry, c.name) + " Inputs: " + str(input_types(c)) for c in capabilities]
    descriptions.append("None: unsupported, ambiguous or insufficient information; ask for clarification.")
    records = []
    for dataset in args.evaluation:
        for case in json.loads(Path(dataset).read_text())["cases"]:
            started = monotonic()
            intent = case["intent"]
            paths, numbers = literals(intent)
            widths = [n for n in numbers if 1 <= n <= 32] or [16]
            questions = {"capability": choice("Which stored function performs the requested operation? Choose None if no function fits.", descriptions),
                         "width": choice("Which number is explicitly the register bit width?", widths)}
            first = model.predict(intent, questions)
            ci = int(first["answers"]["capability"]["choice"])
            width = widths[int(first["answers"]["width"]["choice"])]
            record = {"dataset": dataset, "name": case["name"], "intent": intent,
                      "first_questions": questions, "first_decision": first, "width": width}
            try:
                if ci == len(capabilities):
                    record["status"] = "needs_input"
                    record["passed"] = case["expect"]["status"] != "executed"
                else:
                    cap = capabilities[ci]
                    kinds, roles = input_types(cap), argument_roles(cap)
                    options = [paths if t == "path" else numbers if t == "value" else [] for t in kinds]
                    record["capability"] = cap.name
                    if any(not values for values in options):
                        record["status"] = "needs_input"
                        record["passed"] = case["expect"]["status"] != "executed"
                    else:
                        q = {f"x{i}": choice(f"For this function: {descriptions[ci]} Select input {i}: {roles[i]}. Copy the user's intended argument, not the register bit width.", values)
                             for i, values in enumerate(options)}
                        second = model.predict(intent, q)
                        operands = [values[int(second["answers"][f"x{i}"]["choice"])] for i, values in enumerate(options)]
                        portable = [{"utf8": value} if kind == "path" else value for value, kind in zip(operands, kinds)]
                        record.update({"binding_questions": q, "binding_decision": second, "operands": [portable]})
                        host = MemoryHostContext({n: bytes.fromhex(v) for n, v in case["before"].items()})
                        isolated = VectorRuntime.load(args.program, host=host)
                        rows = convert([portable], host)
                        response = isolated.request(cap.name, rows, width)
                        record.update({"status": response.status, "response": asdict(response),
                                       "files": {n: v.hex() for n, v in host.files.items()}})
                        expected = case["expect"]
                        checks = {"status": expected["status"] == "executed", "output": response.outputs == expected.get("outputs"),
                                  "files": record["files"] == expected["files"],
                                  "binding": {"width": width, "operands": [portable]} == expected.get("request")}
                        record.update({"checks": checks, "passed": all(checks.values())})
            except Exception as error:
                record.update({"error": repr(error), "passed": False})
            record["seconds"] = monotonic() - started
            records.append(record)
            print(json.dumps({k: record.get(k) for k in ("name", "passed", "capability", "width", "operands", "error", "seconds")}, ensure_ascii=False), flush=True)
    summary = {"protocol": "replay; all decisions from real Laya; memory-only execution, no teaching or native effects; expected answers not supplied to model",
               "model": manifest, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "passed": sum(r["passed"] for r in records), "total": len(records), "cases": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": summary["passed"], "total": summary["total"]}))


if __name__ == "__main__":
    main()
