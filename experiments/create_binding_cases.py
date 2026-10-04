"""Freeze new binding requests before inference; expected bindings are scoring only."""
import json
from pathlib import Path

from create_catalog_agent_cases import case, numeric


def main():
    xor = numeric(4, [[3, 6], [9, 4]], lambda a, b: a ^ b)
    sub = numeric(4, [[12, 5], [14, 3]], lambda a, b: a - b)
    cases = []
    for name, intent, a, b, width, operation, evidence in [
        ("xor_width8", "Use an 8-bit register to XOR 255 with 129. Use a stored function; no learning.", 255, 129, 8, lambda a,b:a^b, xor),
        ("xor_korean8", "119와 54를 비트별 XOR로 계산해줘. 실행 폭은 8비트, 학습은 하지 마.", 119, 54, 8, lambda a,b:a^b, xor),
        ("subtract_roles", "Subtract 37 from 313 using a 16-bit register. Use a stored function, without teaching.", 313, 37, 16, lambda a,b:a-b, sub),
        ("subtract_korean", "31000에서 701을 빼줘. 16비트로 실행하고 새로 학습하지 마.", 31000, 701, 16, lambda a,b:a-b, sub)]:
        item = case(name, intent, [operation(a,b)], evidence=evidence)
        item["expect"]["request"] = {"width": width, "operands": [[a,b]]}
        cases.append(item)
    for name, intent, path, value, before, after, evidence in [
        ("map_binding", "XOR each byte of image.bin with 19. Save to image.bin using width 16. Keep spare.bin unchanged; do not learn.", "image.bin", 19,
         {"image.bin":"0080ff42", "spare.bin":"1100"}, {"image.bin":"1393ec51", "spare.bin":"1100"},
         {"cases":[{"inputs":[{"utf8":"example.bin"},3],"before":{"example.bin":"0010"},"after":{"example.bin":"0313"},"output":2}]}),
        ("fill_binding", "결과.bin의 모든 바이트를 값 23으로 채워서 같은 파일에 저장해줘. 실행 폭은 16비트, 학습하지 마.", "결과.bin", 23,
         {"결과.bin":"00ee12"}, {"결과.bin":"171717"},
         {"cases":[{"inputs":[{"utf8":"example.bin"},4],"before":{"example.bin":"0077"},"after":{"example.bin":"0404"},"output":2}]})]:
        item = case(name, intent, [len(bytes.fromhex(before[path]))], before=before, after=after, evidence=evidence)
        item["expect"]["request"] = {"width":16, "operands":[[{"utf8":path},value]]}
        cases.append(item)
    cases.extend([
        case("unsupported_binding", "Multiply 13 by 9 at width 16. Do not learn; ask for learning evidence if unsupported.", None,
             evidence=numeric(4,[[2,4],[3,5]],lambda a,b:a*b),status="unsupported"),
        case("missing_goal", "Inspect task.bin later; I have not chosen what operation or result I want. Ask me for the missing goal.", None,
             before={"task.bin":"00ffa1"},status="clarify")])
    data = {"protocol":"new fixed binding requests frozen before first inference; no retuning on this evaluation", "cases":cases}
    path = Path("experiments/requests/gemma_binding_first_use.json")
    with path.open("x",encoding="utf-8") as stream:
        stream.write(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
    print("Frozen 8 new requests with output, full snapshot, exact binding and terminal-status checks.")


if __name__ == "__main__":
    main()
