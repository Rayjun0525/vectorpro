"""Freeze data-only agent tests before inference; Python oracles are scoring only.

Caller examples are explicitly labelled test inputs, not model-generated evidence.
Self-evidence cases receive neither oracle answers nor lessons from this script.
"""
import json
from pathlib import Path
import random


def numeric(width, rows, operation):
    return {"width": width, "operands": rows, "targets": [operation(*r) for r in rows]}


def case(name, intent, expected, *, evidence=None, before=None, after=None, status="executed", learning=False):
    before = before or {}
    result = {"name": name, "intent": intent, "before": before,
              "expect": {"status": status, "files": before if after is None else after}, "allow_learning": learning}
    if expected is not None:
        result["expect"]["outputs"] = expected
    if evidence is not None:
        result["evidence"] = evidence
    return result


def nand_case(name, a, b, learned):
    result = case(name, f"Learn a NEW W-bit bitwise NAND function named {learned}: complement of AND, unsigned W-bit output. "
                  f"Use your own distinct 4-bit training and validation examples. Then compute it on {a} and {b} at width 8. "
                  "Do not ask me for examples. Use actual tools, save the learned function and report the result.",
                  [(~(a & b)) & 255], learning=True)
    rng = random.Random(56)
    rows = [[rng.randrange(65536), rng.randrange(65536)] for _ in range(100)]
    result["expect"].update(learned=learned, heldout=numeric(16, rows, lambda a, b: (~(a & b)) & 65535))
    return result


def main():
    root = Path("experiments/requests")
    development = {"protocol": "development case; may be inspected during adapter improvements", "cases": [
        nand_case("development_nand", 6, 3, "small_nand")]}
    xor = numeric(4, [[2, 5], [7, 4]], lambda a, b: a ^ b)
    sub = numeric(4, [[10, 3], [13, 6], [15, 4]], lambda a, b: a - b)
    before = {"payload.dat": "00ff7e11", "keep.dat": "aa"}
    file_evidence = {"cases": [{"inputs": [{"utf8": "sample.dat"}, 2],
        "before": {"sample.dat": "1020", "untouched.dat": "aa"},
        "after": {"sample.dat": "1222", "untouched.dat": "aa"}, "output": 2}]}
    fill_evidence = {"cases": [{"inputs": [{"utf8": "sample.dat"}, 5],
        "before": {"sample.dat": "0010ff"}, "after": {"sample.dat": "050505"}, "output": 3}]}
    heldout = {"protocol": "first-use fixed requests, operands and caller examples; frozen before inference after development; do not retune on these results",
        "cases": [
            case("xor_new_english", "Return the bitwise exclusive-OR of 73 and 42 as an 8-bit value. Use stored functions; do not teach.", [73 ^ 42], evidence=xor),
            case("xor_new_korean", "2024와 197을 비트별 XOR로 계산해 줘. 폭은 16비트야. 새로 학습하지 마.", [2024 ^ 197], evidence=xor),
            case("sub_new_english", "Take 271 away from 10490 with register width 16. Use a verified stored function; do not learn.", [10490 - 271], evidence=sub),
            case("sub_new_korean", "950에서 123을 빼줘. 16비트로 실행하고 학습은 하지 마.", [950 - 123], evidence=sub),
            case("map_new_file", "XOR every byte in payload.dat with the value 11, write back to payload.dat at width 16. Leave keep.dat unchanged. Do not teach.", [4],
                 evidence=file_evidence, before=before, after={"payload.dat": bytes(b ^ 11 for b in bytes.fromhex(before["payload.dat"])).hex(), "keep.dat": "aa"}),
            case("fill_new_file", "Replace every byte of data.bin with the constant value 9. Save to data.bin, width 16. No learning.", [3],
                 evidence=fill_evidence, before={"data.bin": "0044aa"}, after={"data.bin": "090909"}),
            case("unsupported_product", "Calculate the product of 11 and 7 with width 16. Do not learn or use an unrelated function. Ask for learning evidence if unsupported.", None,
                 evidence=numeric(4, [[2, 3], [4, 2]], lambda a, b: a * b), status="unsupported"),
            case("unsupported_delete", "자료.bin 파일을 삭제해 줘. 폭은 16비트야. 지원하지 않으면 학습에 필요한 근거를 요청하고, 다른 동작으로 대체하지 마.", None,
                 evidence={"cases": [{"inputs": [{"utf8": "sample.dat"}], "before": {"sample.dat": "01ff"}, "after": {}}]},
                 before={"자료.bin": "01ff"}, status="unsupported"),
            case("unsupported_square", "Square the number 123 at width 16. Do not teach; ask for learning examples if there is no matching function.", None,
                 evidence=numeric(4, [[2], [3]], lambda a: a * a), status="unsupported"),
            case("ambiguous_new_file", "Please process note.dat. I have not selected an operation or a desired result. Ask what I need before acting.", None,
                 before={"note.dat": "abcdef"}, status="clarify"),
            case("self_xor_new", "Compute bitwise XOR of 129 and 37 at width 8. Prepare two distinct correct small validation cases yourself and execute a verified stored function. Do not teach.", [129 ^ 37]),
            nand_case("self_nand_new", 12, 10, "independent_nand")
        ]}
    for name, data in (("catalog_agent_development_v2.json", development), ("catalog_agent_holdout_v2.json", heldout)):
        path = root / name
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print("Frozen development and first-use evaluation files; no model inference or teaching performed.")


if __name__ == "__main__":
    main()
