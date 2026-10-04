"""Freeze new width-binding cases; reference transforms are scoring data only."""
import copy
import json
from pathlib import Path


def main():
    source = json.loads(Path("experiments/requests/gemma_paths_first_use.json").read_text())["cases"]
    specs = [
        ("fill_ko_width24", "‘새 폴더/시험.bin’의 모든 바이트를 21로 채워줘. 보관.dat은 그대로 두고 레지스터 폭은 24비트야. 학습하지 마.", "fill", "새 폴더/시험.bin", 21, 24),
        ("fill_en_register", "Fill all bytes of 'fresh bytes.bin' with 19. Keep 보관.dat unchanged. Use a 16-bit register. Do not learn.", "fill", "fresh bytes.bin", 19, 16),
        ("fill_ko_equal", "새자료.bin을 읽고 모든 바이트를 32로 채워 저장해줘. 보관.dat은 유지하고 폭=16비트로 실행해. 학습은 하지 마.", "fill", "새자료.bin", 32, 16),
        ("fill_en_width32", "Replace every byte of folder/new2026.bin with 15 and save. Retain 보관.dat. Register width is 32 bits. No learning.", "fill", "folder/new2026.bin", 15, 32),
        ("xor_ko_width24", "새변환.bin의 각 바이트에 13을 XOR하고 같은 파일에 저장해줘. 보관.dat은 그대로 두고 폭은 24비트야. 학습하지 마.", "map", "새변환.bin", 13, 24),
        ("xor_en_width16", "XOR every byte of 'new folder/test bytes.bin' with mask 23, then save back. Leave 보관.dat unchanged. Width: 16 bits. No learning.", "map", "new folder/test bytes.bin", 23, 16),
    ]
    cases = []
    for name, intent, operation, path, value, width in specs:
        case = copy.deepcopy(source[0 if operation == "fill" else 2])
        before = bytes.fromhex("0017ff62")
        after = bytes([value] * len(before)) if operation == "fill" else bytes(b ^ value for b in before)
        case.update(name=name, intent=intent, before={path:before.hex(), "보관.dat":"aa55"})
        case["expect"] = {"status":"executed", "files":{path:after.hex(), "보관.dat":"aa55"},
                          "outputs":[4], "request":{"width":width, "operands":[[{"utf8":path}, value]]}}
        cases.append(case)
    with Path("experiments/requests/gemma_width_first_use.json").open("x", encoding="utf-8") as output:
        json.dump({"protocol":"new width-binding requests frozen before first inference; caller evidence supplied, expected files never supplied to model", "cases":cases}, output, ensure_ascii=False, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
