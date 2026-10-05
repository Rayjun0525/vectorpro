"""Optional semantic catalog: tensor-derived documents, typed candidates and evidence.

Similarity is candidate ranking, never authorization or proof of user intent.
"""
import json
import hashlib
from time import monotonic
import torch
import torch.nn.functional as F
from vectorpro.host import HOST_TYPES, MemoryHostContext
from vectorpro.learning.registry import Registry
from vectorpro.machine import VectorProgram
from vectorpro.tensor_codec import encode_tree, decode_tree

HOST_TEXT = {
    "process.pipeline": "Execute concurrent processes joined by OS pipes and retain streams and exit codes.",
    "network.http": "Send a bounded HTTP request and retain status and body bytes.",
    "network.body": "Extract HTTP body bytes, including error responses.",
    "network.require_body": "Extract body bytes only for a successful HTTP status; otherwise raise an error.",
    "network.status": "Extract the HTTP response status.",
    "network.ok": "Return 1 for successful HTTP status and 0 otherwise.",
    "system.info": "Query platform, machine, CPU count, host cwd or process ID as JSON bytes.",
    "process.run": "Execute argv with input bytes and return captured streams and exit code.",
    "process.stdout": "Extract stdout bytes from a process result.",
    "process.stderr": "Extract stderr bytes from a process result.",
    "process.code": "Extract the process exit code; signals map to 128 plus signal.",
    "record.pack": "Pack a byte buffer and unsigned value into a portable binary record.",
    "record.buffer": "Extract the byte buffer from a portable binary record.",
    "record.value": "Extract the unsigned value from a portable binary record.",
    "text.ends_with": "Check whether UTF-8 text ends with a supplied suffix; return 1 or 0. 문자열 접미사 조건 확인.",
    "list.length": "Count items of a NUL-terminated byte list. 목록 항목 개수.",
    "list.get": "Extract one byte-list item by zero-based index. 목록 항목 추출.",
    "list.append": "Return a new byte list with an appended item. 목록 항목 추가.",
    "path.join": "Join a relative parent path and a relative UTF-8 child buffer. 상대 경로 결합.",
    "text.concat": "Concatenate two valid UTF-8 texts. 문자열 결합.",
    "json.get": "Read an object field or array index from JSON bytes and return JSON bytes. 구조화 필드 조회.",
    "json.text": "Decode a JSON string as UTF-8 bytes. JSON 문자열 추출.",
    "buffer.new": "Allocate, create a new byte array in memory with the requested size. 지정한 길이의 새 메모리 바이트 버퍼 생성 할당.",
    "buffer.length": "Measure buffer size, count how many bytes a memory buffer contains. 메모리 버퍼 길이 크기 바이트 개수 확인.",
    "buffer.get": "Read, retrieve, fetch one byte from a memory buffer at a position or offset. 버퍼의 특정 인덱스 위치에서 바이트 하나 읽기.",
    "buffer.set": "Write, replace, update one byte in a memory buffer at a position or offset. 메모리 버퍼의 특정 위치에 바이트 하나 쓰기 변경.",
    "file.read": "Read, load file contents from disk into memory, returning a byte buffer. 디스크 파일 내용을 읽어 메모리 버퍼로 불러오기.",
    "file.write": "Write, save buffer contents from memory to a disk file; return bytes written. 메모리 버퍼 내용을 디스크 파일에 저장 쓰기.",
    "path.exists": "Check whether a file or directory exists; return 1 or 0. 파일 디렉터리 존재 확인.",
    "file.remove": "Remove one file; return 1. 파일 하나 삭제.",
    "file.move": "Move a file to a destination path, replacing a destination file; return 1. 파일 이동 이름 변경.",
    "directory.create": "Create one directory with an existing parent; return 1. 디렉터리 생성.",
    "directory.remove": "Remove one empty directory; return 1. 빈 디렉터리 삭제.",
    "directory.list": "Read sorted immediate directory entry names as UTF-8 bytes, each terminated by NUL. 디렉터리 항목 목록 조회.",
}


def document(registry, name, visited=()):
    cap = registry.get(name)
    if name in visited:
        return "recursive dependency"
    prov = cap.provenance
    if prov["kind"] == "host":
        return HOST_TEXT[prov["operation"]].split(". ")[0] + "."
    if prov["kind"] == "unit":
        text = cap.plan.description
        if prov["signature"] == [2, 0, 1] and prov["table"] == [[0], [1], [1], [0]]:
            text = "Bitwise exclusive OR (XOR) of two integers; bits differ."
        if prov["signature"] == [2, 1, 1] and prov["table"] == [[0, 0], [1, 0], [1, 1], [0, 0], [1, 1], [0, 0], [0, 1], [1, 1]]:
            text = "Subtract the second number from the first number; integer difference with borrow."
        return text
    program = VectorProgram.from_data(prov["program"])
    registers = ["constant " + str(int(row.argmax())) for row in program.init]
    for i, row in enumerate(program.inputs):
        types = prov.get("input_types", ["value"] * cap.plan.arity)
        registers[int(row.argmax())] = "input " + types[i]
    details, mutations = [], []
    for step in range(program.n_steps):
        if not bool(program.calls[step]):
            continue
        called = registry.get(registry.name_of(program.keys[step]))
        args = [registers[int(row.argmax())] for row in program.reads[step][:called.plan.arity]]
        kind = called.provenance["kind"]
        if kind == "host":
            operation = called.provenance["operation"]
            if operation == "buffer.get":
                expr = "byte from " + args[0]
            elif operation == "buffer.set":
                expr = "write " + args[2] + " into " + args[0] + " at an index"
                if args[2] == "input value":
                    mutations.append("Fill all bytes of a file with the same supplied value.")
                else:
                    mutations.append("Transform every byte of a file using " + args[2] + ".")
            elif operation == "file.read":
                expr = "file bytes read from " + args[0]
            elif operation == "file.write":
                expr = "save " + args[1] + " to " + args[0]
            else:
                expr = HOST_TEXT[operation]
        elif kind == "unit":
            expr = document(registry, called.name) + " Applied to " + " and ".join(args)
            if called.provenance["signature"] == [2, 0, 1] and called.provenance["table"] == [[0], [1], [1], [0]]:
                expr = "bitwise exclusive OR (XOR) of " + " and ".join(args)
        else:
            expr = document(registry, called.name, (*visited, name))
            mutations.append(expr)
        details.append(expr)
        destination = int(program.writes[step].argmax())
        if destination < len(registers):
            registers[destination] = expr
    has_guard = bool((program.cond.argmax(dim=1) < program.n_registers).any())
    prefix = []
    if prov.get("control") == "list-iteration":
        prefix.append("Iterate over list items using acquired argument routing and calls.")
    elif prov.get("control") == "list-selection":
        prefix.append("Conditionally execute acquired calls for selected list items.")
    elif prov.get("control") == "list-reduction":
        prefix.append("Accumulate list results using acquired arithmetic and routing.")
    elif program.has_loop:
        prefix.append("Repeat indexed operations over the bytes in a file buffer.")
    elif has_guard:
        prefix.append("Execute acquired calls according to stored conditional routing."
                      if prov.get("branch_description_version") == 2
                      else "Conditionally execute when a flag is nonzero; otherwise skip.")
    if prov.get("input_types", []).count("path") > 1:
        prefix.append("Apply operations to two separate files.")
    return " ".join(prefix + list(dict.fromkeys(mutations or details)))


class Encoder:
    def __init__(self, path):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True).eval()

    def __call__(self, texts):
        tensors = []
        for start in range(0, len(texts), 8):
            inputs = self.tokenizer(texts[start:start + 8], padding=True, truncation=True,
                                    max_length=128, return_tensors="pt")
            with torch.inference_mode():
                hidden = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                tensors.append(F.normalize(pooled, dim=1))
        return torch.cat(tensors)


def input_types(cap):
    if cap.provenance["kind"] == "host":
        return list(HOST_TYPES[cap.provenance["operation"]][0])
    return cap.provenance.get("input_types", ["value"] * cap.plan.arity)


def argument_roles(cap):
    roles = [("relative file path" if t == "path" else "hexadecimal buffer bytes" if t == "buffer" else "numeric parameter")
             for t in input_types(cap)]
    p = cap.provenance
    if (p["kind"] == "unit" and p.get("signature") == [2, 1, 1]
            and p.get("table") == [[0, 0], [1, 0], [1, 1], [0, 0], [1, 1], [0, 0], [0, 1], [1, 1]]
            and p.get("schema") == {"kind": "scan", "arity": 2, "initial_state": [0],
                                   "direction": "lsb_first", "emit_final_state": True}):
        roles = ["minuend: number to subtract FROM (left operand)",
                 "subtrahend: number to subtract (right operand)"]
    return roles


def fingerprint(registry):
    encoded = json.dumps(registry.to_data(), sort_keys=True, separators=(",", ":")).encode()
    return torch.tensor(list(hashlib.sha256(encoded).digest()), dtype=torch.uint8)


def operand_types(rows):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 64:
        raise ValueError("provide 1..64 portable operand rows")
    shapes = []
    for row in rows:
        if not isinstance(row, list) or not 1 <= len(row) <= 3:
            raise ValueError("provide 1..3 operands per row")
        types = []
        for value in row:
            if type(value) is int and value >= 0:
                types.append("value")
            elif isinstance(value, dict) and set(value) == {"utf8"} and isinstance(value["utf8"], str):
                MemoryHostContext.normalize(value["utf8"])
                types.append("path")
            elif isinstance(value, dict) and set(value) == {"hex"} and isinstance(value["hex"], str):
                if len(value["hex"]) > 131072:
                    raise ValueError("buffer operand exceeds catalog limit")
                bytes.fromhex(value["hex"])
                types.append("buffer")
            else:
                raise ValueError("use JSON integers, {utf8:relative_path}, or {hex:bytes}; strings/bools/raw handles are not typed values")
        shapes.append(types)
    if any(shape != shapes[0] for shape in shapes):
        raise ValueError("operand rows must have identical types")
    return shapes[0]


def convert(rows, host):
    operand_types(rows)
    result = []
    for row in rows:
        values = []
        for value in row:
            if isinstance(value, dict):
                if host is None:
                    raise ValueError("portable buffers require a host")
                raw = value["utf8"].encode() if "utf8" in value else bytes.fromhex(value["hex"])
                value = host.put(raw)
            values.append(value)
        result.append(tuple(values))
    return result


class TensorCatalog:
    """Search/verify against the current registry; optional pretrained encoder."""
    def __init__(self, runtime, encoder, identity=None):
        self.runtime, self.encoder = runtime, encoder
        self.identity = identity or {}
        self.refresh()

    def refresh(self):
        digest = fingerprint(self.runtime.registry)
        if hasattr(self, "digest") and torch.equal(self.digest, digest):
            return
        self.names = [c.name for c in self.runtime.registry]
        self.documents = [document(self.runtime.registry, name) for name in self.names]
        # One authoritative contract; compact projection keeps small-model prompts bounded.
        fields = ("id", "version", "name", "input_types", "argument_roles", "output_width", "has_effects")
        self.contracts = [{k: contract[k] for k in fields} for contract in self.runtime.contracts()]
        extras = self.runtime._tensor_extras
        try:
            meta = decode_tree({k: extras["catalog_" + k] for k in ("nodes", "bytes", "floats")})
            reusable = (meta["names"] == self.names and meta["documents"] == self.documents
                        and meta["contracts"] == self.contracts and meta["encoder"] == self.identity)
        except (KeyError, ValueError, TypeError, IndexError):
            reusable = False
        if reusable:
            self.vectors = extras["semantic_vectors"]
        else:
            self.vectors = self.encoder(self.documents) if self.names else torch.empty((0, 0))
        if self.names and (self.vectors.ndim != 2 or len(self.vectors) != len(self.names)
                           or not bool(torch.isfinite(self.vectors).all())):
            raise ValueError("invalid semantic catalog tensor")
        self.digest = digest
        self.runtime._tensor_extras = {"registry_digest": digest, "semantic_vectors": self.vectors,
            **{"catalog_" + k: v for k, v in encode_tree({"names": self.names,
               "documents": self.documents, "contracts": self.contracts, "encoder": self.identity,
               "pooling": "masked_mean", "max_length": 128}).items()}}

    def search(self, query, rows, limit=5):
        self.refresh()
        if not isinstance(query, str) or not 1 <= len(query) <= 4000:
            raise ValueError("query must contain 1..4000 characters")
        shape = operand_types(rows)
        if not self.names:
            return []
        scores = self.vectors @ self.encoder([query])[0]
        compatible = [i for i, contract in enumerate(self.contracts) if contract["input_types"] == shape]
        compatible.sort(key=lambda i: float(scores[i]), reverse=True)
        return [{**self.contracts[i], "description": self.documents[i], "score": float(scores[i])}
                for i in compatible[:limit]]

    def search_goal(self, query, limit=5, types=None):
        self.refresh()
        if not isinstance(query, str) or not 1 <= len(query) <= 4000:
            raise ValueError("query must contain 1..4000 characters")
        if not self.names:
            return []
        scores = self.vectors @ self.encoder([query])[0]
        order = scores.argsort(descending=True).tolist()
        if types is not None:
            order = [i for i in order if self.contracts[i]["input_types"] == types]
        order = order[:limit]
        return [{**self.contracts[i], "description": self.documents[i], "score": float(scores[i])} for i in order]

    def resolve(self, query, rows, width, numeric_validation=None, state_validation=None):
        if type(width) is not int or not 1 <= width <= 32:
            raise ValueError("catalog requests require width 1..32")
        shape = operand_types(rows)
        if any(type(v) is int and v >= 1 << width for row in rows for v in row):
            raise ValueError("request integers must fit the unsigned register width")
        candidates = self.search(query, rows, limit=32)
        response = {"input_types": shape, "candidates": candidates}
        if not candidates:
            return {**response, "status": "needs_learning_examples", "reason": "no compatible stored capability"}
        if numeric_validation is not None and state_validation is not None:
            raise ValueError("provide numeric_validation OR state_validation")
        if numeric_validation is None and state_validation is None:
            return {**response, "status": "needs_evidence", "reason": "similarity alone cannot verify intent; provide numeric examples or complete before/after snapshots"}
        if numeric_validation is not None:
            if shape != ["value"] * len(shape):
                raise ValueError("numeric validation requires value operands")
            from vectorpro.learning.examples import ExampleSet
            if set(numeric_validation) != {"width", "operands", "targets"}:
                raise ValueError("numeric validation needs width/operands/targets")
            examples = ExampleSet(numeric_validation["width"],
                                  [tuple(r) for r in numeric_validation["operands"]],
                                  list(numeric_validation["targets"]))
            if not 2 <= len(examples.operands) <= 32:
                raise ValueError("numeric validation needs 2..32 distinct examples")
            if type(examples.width) is not int or not 1 <= examples.width <= 32:
                raise ValueError("validation width must be 1..32")
            if len(set(examples.operands)) != len(examples.operands) or len(examples.targets) != len(examples.operands):
                raise ValueError("validation examples need distinct inputs and one target each")
            if any(type(x) is not int or not 0 <= x < 1 << examples.width for row in examples.operands for x in row):
                raise ValueError("validation operands must fit their width")
            if any(type(t) is not int or not 0 <= t < 1 << (2 * examples.width) for t in examples.targets):
                raise ValueError("validation targets must be bounded unsigned integers")
            if operand_types([list(r) for r in examples.operands]) != shape:
                raise ValueError("validation types do not match requested operands")
        else:
            if not isinstance(state_validation, list) or not 1 <= len(state_validation) <= 8:
                raise ValueError("state validation needs 1..8 complete file snapshots")
            if len(rows) != 1:
                raise ValueError("state requests require one execution lane")
            for case in state_validation:
                if set(case) - {"inputs", "before", "after", "output", "before_directories", "after_directories"} or not {"inputs", "before", "after"} <= set(case):
                    raise ValueError("state evidence accepts only inputs/before/after/output")
                if operand_types([case["inputs"]]) != shape:
                    raise ValueError("state validation types do not match requested operands")
                if any(type(v) is int and v >= 1 << width for v in case["inputs"]):
                    raise ValueError("state validation integers must fit the requested width")
                for field, files in (("before_directories", case["before"]), ("after_directories", case["after"])):
                    directories = case.get(field, [])
                    if (not isinstance(directories, list) or len(directories) > 32
                            or any(not isinstance(p, str) or MemoryHostContext.normalize(p) != p for p in directories)
                            or len(set(directories)) != len(directories)):
                        raise ValueError("directory snapshots need at most 32 distinct normalized paths")
                    MemoryHostContext.directory_state(files, directories)
                for snapshot in (case["before"], case["after"]):
                    if not isinstance(snapshot, dict) or len(snapshot) > 32:
                        raise ValueError("snapshot needs at most 32 files")
                    if sum(len(v) for v in snapshot.values()) > 131072:
                        raise ValueError("snapshot exceeds evidence size limit")
                    for name, value in snapshot.items():
                        MemoryHostContext.normalize(name)
                        try:
                            bytes.fromhex(value)
                        except (ValueError, TypeError):
                            raise ValueError("before/after must map actual filenames directly to HEX bytes, not separate path/hex fields") from None
        accepted = []
        deadline = monotonic() + 60
        for candidate in candidates:
            if monotonic() > deadline:
                return {**response, "status": "needs_evidence", "reason": "verification time budget exhausted"}
            cap = self.runtime.registry.get(candidate["name"])
            # Every verifier has its own registry/host and reads no native files.
            isolated = Registry.from_data(self.runtime.registry.to_data())
            from vectorpro.machine import ProgramExecutable
            for capability in isolated:
                if isinstance(capability.executable, ProgramExecutable):
                    capability.executable.budget = min(capability.executable.budget or 20000, 20000)
            if cap.executable.effects and numeric_validation is not None:
                continue
            try:
                if numeric_validation is not None:
                    outputs = isolated.run(cap.name, examples.operands, examples.width)
                    passed = outputs == examples.targets
                else:
                    passed = True
                    for case in state_validation:
                        context = MemoryHostContext({n: bytes.fromhex(v) for n, v in case["before"].items()}, max_buffer_bytes=65536,
                                                    directories=case.get("before_directories", []))
                        with context.activate():
                            output = isolated.run(cap.name, convert([case["inputs"]], context), width)[0]
                        desired = {n: bytes.fromhex(v) for n, v in case["after"].items()}
                        if (context.files != desired
                                or context.directories != MemoryHostContext.directory_state(desired, case.get("after_directories", []))
                                or ("output" in case and case["output"] != output)):
                            passed = False
                            break
                if passed:
                    accepted.append(candidate["name"])
            except (ValueError, KeyError, OSError, RuntimeError, IndexError):
                continue
        if len(accepted) != 1:
            return {**response, "status": "needs_learning_examples" if not accepted else "needs_evidence",
                    "reason": "no candidate matches evidence" if not accepted else "evidence does not distinguish multiple candidates",
                    "matching_candidates": accepted}
        return {**response, "status": "matched", "name": accepted[0],
                "evidence_source": "caller/LLM supplied, not an independent proof of intent"}
