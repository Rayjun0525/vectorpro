"""Lossless typed tensor tree for runtime data and catalog metadata."""
import torch

def encode_tree(data):
    tape, blob, floats = [], bytearray(), []

    def visit(value):
        if value is None:
            tape.append([0, 0, 0])
        elif isinstance(value, bool):
            tape.append([1, int(value), 0])
        elif isinstance(value, int):
            if -(1 << 63) <= value < 1 << 63:
                tape.append([2, value, 0])
            else:
                raw = value.to_bytes((value.bit_length() + 8) // 8, "little", signed=True)
                tape.append([7, len(blob), len(raw)]); blob.extend(raw)
        elif isinstance(value, float):
            tape.append([3, len(floats), 0]); floats.append(value)
        elif isinstance(value, str):
            raw = value.encode("utf-8")
            tape.append([4, len(blob), len(raw)]); blob.extend(raw)
        elif isinstance(value, (list, tuple)):
            tape.append([5, len(value), 0])
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            tape.append([6, len(value), 0])
            for key, child in value.items():
                if not isinstance(key, str):
                    raise ValueError("tree object keys must be strings")
                visit(key); visit(child)
        else:
            raise ValueError(f"unsupported node {type(value)}")
    visit(data)
    return {"nodes": torch.tensor(tape, dtype=torch.int64),
            "bytes": torch.tensor(list(blob), dtype=torch.uint8),
            "floats": torch.tensor(floats, dtype=torch.float64)}


def decode_tree(tree):
    if (tree["nodes"].dtype != torch.int64 or tree["nodes"].ndim != 2 or tree["nodes"].shape[1] != 3
            or tree["bytes"].dtype != torch.uint8 or tree["bytes"].ndim != 1
            or tree["floats"].dtype != torch.float64 or tree["floats"].ndim != 1):
        raise ValueError("invalid typed tensor tree shape/dtype")
    tape = tree["nodes"].tolist()
    blob = bytes(tree["bytes"].tolist())
    floats = tree["floats"].tolist()
    cursor = 0

    def read():
        nonlocal cursor
        if cursor >= len(tape):
            raise ValueError("truncated tensor tree")
        tag, value, length = tape[cursor]; cursor += 1
        if tag == 0: return None
        if tag == 1:
            if value not in (0, 1): raise ValueError("invalid boolean node")
            return bool(value)
        if tag == 2: return value
        if tag == 3:
            if not 0 <= value < len(floats): raise ValueError("invalid float offset")
            return floats[value]
        if tag == 4:
            if value < 0 or length < 0 or value + length > len(blob): raise ValueError("invalid byte range")
            return blob[value:value + length].decode("utf-8")
        if tag == 7:
            if value < 0 or length < 1 or value + length > len(blob): raise ValueError("invalid integer byte range")
            return int.from_bytes(blob[value:value + length], "little", signed=True)
        if tag in (5, 6) and not 0 <= value <= len(tape) - cursor:
            raise ValueError("invalid container length")
        if tag == 5: return [read() for _ in range(value)]
        if tag == 6:
            result = {}
            for _ in range(value):
                key = read()
                if not isinstance(key, str) or key in result:
                    raise ValueError("object keys must be unique strings")
                result[key] = read()
            return result
        raise ValueError("unknown tensor node type")
    result = read()
    if cursor != len(tape):
        raise ValueError("trailing tensor nodes")
    return result
