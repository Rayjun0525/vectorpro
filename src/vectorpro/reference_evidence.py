"""Caller-installed external references; generic observation, never task recipes.

Temporary roots isolate fixture state, not arbitrary executables at OS level.
Reference selection is a model declaration, not independent intent verification.
"""
import copy
import hashlib
import json
import tempfile
from pathlib import Path

from vectorpro.acquisition import EvidenceBank, digest
from vectorpro.contract_learning import validate_draft
from vectorpro.host import HostContext, MemoryHostContext
from vectorpro.process_host import run, decode


def file_digest(path):
    with open(path, "rb") as stream:
        result = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def snapshot(root):
    files, directories, total = {}, [], 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("reference snapshots do not follow symlinks")
        name = path.relative_to(root).as_posix()
        if path.is_dir():
            directories.append(name)
        elif path.is_file():
            total += path.stat().st_size
            if total > 65536:
                raise ValueError("reference state exceeds byte limit")
            files[name] = path.read_bytes().hex()
        else:
            raise ValueError("unsupported reference filesystem entry")
        if len(files) > 32 or len(directories) > 32:
            raise ValueError("reference state exceeds entry limit")
    return files, directories


class ReferenceProviders:
    def __init__(self, manifests):
        if not isinstance(manifests, list) or len(manifests) > 16:
            raise ValueError("provide at most 16 reference manifests")
        self._providers = {}
        for original in manifests:
            data = copy.deepcopy(original)
            required = {"id", "description", "interface", "argv", "seed_files", "directories", "return", "max_steps"}
            if not isinstance(data, dict) or set(data) != required:
                raise ValueError("reference requires identity/interface/argv/fixture/return/search bound")
            if any(not isinstance(data[k], str) or not 1 <= len(data[k]) <= 500 for k in ("id", "description")) or data["id"] in self._providers:
                raise ValueError("invalid or duplicate reference identity")
            if not isinstance(data["interface"], dict) or set(data["interface"]) != {"parameters", "output", "allowed_operations"}:
                raise ValueError("reference interface needs parameters/output/allowed_operations")
            validate_draft({"version": 1, "name": "reference", "description": data["description"], **data["interface"]})
            params = data["interface"]["parameters"]
            if any(p["type"] != "path" for p in params) or data["interface"]["output"] != {"type": "value", "width": "W"}:
                raise ValueError("reference v1 supports path inputs and W-bit scalar output")
            names = {p["name"] for p in params}
            argv = data["argv"]
            if not isinstance(argv, list) or not 1 <= len(argv) <= 32 or any(not isinstance(a, str) or not a or len(a) > 500 or "\0" in a for a in argv):
                raise ValueError("invalid reference argv")
            executable = Path(argv[0]).resolve(strict=True)
            if not Path(argv[0]).is_absolute() or not executable.is_file():
                raise ValueError("reference executable must be an existing absolute file")
            for token in argv[1:]:
                if "$" in token and (not token.startswith("$") or token[1:] not in names):
                    raise ValueError("argv accepts only whole parameter placeholders")
            seeds = data["seed_files"]
            if not isinstance(seeds, dict) or len(seeds) > 16:
                raise ValueError("reference fixture needs at most 16 seed files")
            for path, value in seeds.items():
                self._path(path, names)
                if value != "$payload" and (not isinstance(value, str) or len(value) % 2 or any(c not in "0123456789abcdefABCDEF" for c in value) or len(value) > 8192):
                    raise ValueError("seed file needs byte hex or $payload")
            dirs = data["directories"]
            if not isinstance(dirs, list) or len(dirs) > 16 or any(not isinstance(d, str) or MemoryHostContext.normalize(d) != d for d in dirs):
                raise ValueError("invalid fixture directories")
            returned = data["return"]
            if returned != {"kind": "success"} and not (isinstance(returned, dict) and set(returned) == {"kind", "parameter"} and returned["kind"] == "file_length" and returned["parameter"] in names):
                raise ValueError("return observation must be success or parameter file_length")
            if type(data["max_steps"]) is not int or not 1 <= data["max_steps"] <= 12:
                raise ValueError("reference search bound must be 1..12")
            self._providers[data["id"]] = (data, executable, file_digest(executable))

    @staticmethod
    def _path(token, names):
        if not isinstance(token, str):
            raise ValueError("fixture paths must be strings")
        if token.startswith("$"):
            if token[1:] not in names:
                raise ValueError("unknown fixture parameter")
        elif MemoryHostContext.normalize(token) != token:
            raise ValueError("fixture paths must be normalized")

    def describe(self):
        return [{"id": data["id"], "description": data["description"],
                 "interface": copy.deepcopy(data["interface"]), "executable_sha256": sha}
                for data, _, sha in self._providers.values()]

    def collect(self, provider_id, intent):
        if not isinstance(provider_id, str) or provider_id not in self._providers:
            raise ValueError("unknown caller-installed reference")
        if not isinstance(intent, str) or not 1 <= len(intent) <= 2000:
            raise ValueError("reference collection needs a bounded intent")
        data, executable, sha = self._providers[provider_id]
        if file_digest(executable) != sha:
            raise ValueError("reference executable changed since registration")
        payloads = (b"abc", b"\0\xff", b"", b"validation", bytes(range(256)), "새 근거".encode(), b"ab" * 4096)
        examples = []
        for index, payload in enumerate(payloads):
            inputs = {p["name"]: f"p{n}-{index}.dat" for n, p in enumerate(data["interface"]["parameters"])}
            substitute = lambda token: inputs[token[1:]] if token.startswith("$") else token
            with tempfile.TemporaryDirectory(prefix="vectorpro-reference-") as temporary:
                root = Path(temporary)
                for directory in data["directories"]:
                    (root / directory).mkdir(parents=True, exist_ok=True)
                for path, value in data["seed_files"].items():
                    target = root / substitute(path)
                    if target.exists() and target.is_dir():
                        raise ValueError("fixture file/directory collision")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(payload if value == "$payload" else bytes.fromhex(value))
                before, before_dirs = snapshot(root)
                argv = [str(executable), *(substitute(a) for a in data["argv"][1:])]
                result = decode(run(HostContext(root, max_buffer_bytes=65536), json.dumps({"argv": argv, "timeout": 2}).encode(), b""))
                if result["code"] != 0:
                    raise ValueError("reference failed; no learning evidence accepted")
                after, after_dirs = snapshot(root)
                output = 1
                if data["return"]["kind"] == "file_length":
                    output = len(bytes.fromhex(after[inputs[data["return"]["parameter"]]]))
                examples.append({"inputs": list(inputs.values()), "before": before, "after": after,
                    "before_directories": before_dirs, "after_directories": after_dirs, "output": output})
        record = {"id": "reference:" + digest(data)[:24], "intent": intent,
            "origin": f"caller-installed reference {provider_id}; executable sha256={sha}; manifest sha256={digest(data)}",
            "interface": copy.deepcopy(data["interface"]),
            "state_lesson": {"input_types": ["path"] * len(data["interface"]["parameters"]),
                "training": examples[:2], "validation": examples[2:4], "max_steps": data["max_steps"]},
            "heldout": examples[4:]}
        EvidenceBank([record])  # Validate all observations before exposing any source.
        return record
