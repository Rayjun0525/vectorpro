"""Freeze path-copy evaluation before inference; expected data are scoring only."""
import json
from pathlib import Path

from create_catalog_agent_cases import case


def main():
    fill = {"cases":[{"inputs":[{"utf8":"probe.bin"},5],"before":{"probe.bin":"0010ff"},"after":{"probe.bin":"050505"},"output":3}]}
    xor = {"cases":[{"inputs":[{"utf8":"probe.bin"},3],"before":{"probe.bin":"0010ff"},"after":{"probe.bin":"0313fc"},"output":3}]}
    cases = []
    rows = [
        ('quoted_korean_fill','‘출력 폴더/샘플 문서.bin’의 모든 바이트를 29로 채워서 저장해줘. 유지.bin은 그대로 두고, 폭은 16비트야. 학습하지 마.', '출력 폴더/샘플 문서.bin',29,'0012ff45','유지.bin','fill'),
        ('bare_korean_fill','실험.bin의 바이트를 31로 채워서 같은 파일에 저장해줘. 고정.dat은 유지해. 16비트로 실행하고 학습하지 마.', '실험.bin',31,'00ee12','고정.dat','fill'),
        ('nested_korean_xor','XOR all bytes of raw.v1/블록.bin with 7 and write back. Leave safe.bin unchanged. Use width 16 and no learning.', 'raw.v1/블록.bin',7,'01ff00','safe.bin','xor'),
        ('quoted_english_xor','XOR each byte of "samples/mixed bytes.bin" with 21. Save to that same file; preserve keep.bin. Execute at width 16 without teaching.', 'samples/mixed bytes.bin',21,'00ff4212','keep.bin','xor'),
        ('filename_digits_zero','Fill data2024.bin with byte value 0 using width 16. Keep backup16.bin unchanged. Do not learn.', 'data2024.bin',0,'ff01ee','backup16.bin','fill')]
    for name,intent,path,value,hex_bytes,keep,kind in rows:
        payload = bytes.fromhex(hex_bytes)
        after = bytes([value])*len(payload) if kind=='fill' else bytes(b^value for b in payload)
        item = case(name,intent,[len(payload)],before={path:hex_bytes,keep:'aa55'},after={path:after.hex(),keep:'aa55'},evidence=fill if kind=='fill' else xor)
        item['expect']['request']={'width':16,'operands':[[{'utf8':path},value]]}
        cases.append(item)
    cases.append(case('missing_path','모든 바이트를 17로 채워줘. 대상 파일 경로는 아직 정하지 않았어. 16비트로 실행하고 학습하지 마.',None,
                      before={'keep.bin':'00ff'},evidence=fill,status='clarify'))
    data={'protocol':'new path requests frozen before first inference after development; no retuning on results','cases':cases}
    with Path('experiments/requests/gemma_paths_first_use.json').open('x',encoding='utf-8') as output:
        output.write(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print('Frozen 6 path requests; expected paths and files are scoring only.')


if __name__=='__main__':
    main()
