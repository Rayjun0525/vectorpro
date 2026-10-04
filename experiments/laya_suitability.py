"""Frozen, role-separated Laya suitability diagnostic; no execution or teaching.

Caller-supplied input types narrow contracts. Correct contracts are supplied ONLY
for the separate binding diagnostic, never as the selector's expected answer.
Every case is measured under three candidate orders, with no tuning between runs.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from time import monotonic

from experiments.local_laya import literals
from vectorpro.runtime import VectorRuntime
from vectorpro.semantic_catalog import argument_roles, document, input_types


def cases():
    result = []
    def add(name, intent, types, expected, width=None, operands=None):
        result.append(dict(name=name, intent=intent, types=types, expected=expected,
                           width=width, operands=operands,
                           language="ko" if any("가" <= c <= "힣" for c in intent) else "en"))
    for name, intent, expected, width, operands in [
        ("xor_en1", "XOR the integers 83 and 46 using a 16-bit register.", "xor", 16, [83,46]),
        ("xor_en2", "With register width 8, calculate bitwise exclusive OR of 197 and 58.", "xor", 8, [197,58]),
        ("xor_ko1", "폭 16비트로 정수 71과 38의 비트별 XOR를 구해줘.", "xor", 16, [71,38]),
        ("xor_ko2", "정수 203과 91을 비트별 배타적 논리합으로 계산해줘. 레지스터 폭은 8비트.", "xor", 8, [203,91]),
        ("sub_en1", "Subtract 47 from 182. Register width is 16 bits.", "sub", 16, [182,47]),
        ("sub_en2", "Use 8 bits: the difference of 214 minus 63.", "sub", 8, [214,63]),
        ("sub_ko1", "폭은 16비트로 하고 176에서 59를 빼줘.", "sub", 16, [176,59]),
        ("sub_ko2", "레지스터 폭 8비트. 225에서 74를 뺀 값을 계산해줘.", "sub", 8, [225,74])]:
        add(name,intent,["value","value"],expected,width,operands)
    for name, intent, expected, width, operands in [
        ("fill_en1", "Fill every byte of target.bin with 27. Leave keep.bin alone. Register width: 16 bits.", "fill",16,["target.bin",27]),
        ("fill_en2", "Using a 16-bit register, replace all bytes in 'input records.bin' with 13; retain spare.dat.", "fill",16,["input records.bin",13]),
        ("fill_ko1", "대상.bin의 모든 바이트를 25로 채워줘. 보존.bin은 그대로 두고 레지스터 폭은 16비트.", "fill",16,["대상.bin",25]),
        ("fill_ko2", "폭 16비트로 ‘문서 폴더/자료.bin’ 전체 바이트를 11로 바꿔줘. 남김.dat은 건드리지 마.", "fill",16,["문서 폴더/자료.bin",11]),
        ("map_en1", "XOR each byte in payload.bin with mask 9, and save back. Keep hold.bin unchanged. Register width 16.", "map",16,["payload.bin",9]),
        ("map_en2", "Use 16-bit registers to transform 'archive bytes.bin' by bitwise exclusive OR of every byte with 22. Retain backup.bin.", "map",16,["archive bytes.bin",22]),
        ("map_ko1", "변환.bin의 각 바이트를 마스크 5와 XOR해서 같은 파일에 저장해줘. 보호.bin은 유지해. 폭은 16비트.", "map",16,["변환.bin",5]),
        ("map_ko2", "레지스터 폭 16비트로 ‘임시 폴더/내용.bin’의 바이트마다 18과 배타적 논리합을 수행해줘. 원본.dat은 그대로 둬.", "map",16,["임시 폴더/내용.bin",18])]:
        add(name,intent,["path","value"],expected,width,operands)
    for name,intent,types in [
        ("and_en","Calculate bitwise AND of integers 85 and 39 with width 16.",["value","value"]),
        ("add_ko","16비트 정수 143과 52를 더해줘.",["value","value"]),
        ("mul_en","Multiply 26 by 7 using 16-bit integers.",["value","value"]),
        ("or_ko","16비트 정수 101과 42를 비트별 OR로 계산해줘.",["value","value"]),
        ("append_en","Append 14 bytes to report.bin; do not overwrite existing data. Register width 16.",["path","value"]),
        ("truncate_ko","폭 16비트로 문서.bin의 길이를 6바이트로 줄여줘.",["path","value"]),
        ("rotate_en","Rotate every byte of source.bin left by 3 bits; width 16.",["path","value"]),
        ("increment_ko","폭 16비트로 자료.bin의 각 바이트에 4를 더해서 저장해줘.",["path","value"])]:
        add(name,intent,types,"none")
    return result


def decision(model, state, instructions, options):
    question = {"type":"choice", "instructions":instructions,
                "criteria":dict(options)}
    started = monotonic()
    response = model.predict(state, {"answer":question})
    return {"question":question, "response":response, "seconds":monotonic()-started,
            "selected":response["answers"]["answer"]["choice"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--model",default="/opt/vectorpro-models/laya-multilingual")
    args=parser.parse_args()
    if args.output.exists():
        parser.error("preserve previous results; select a new output directory")
    args.output.mkdir(parents=True)
    dataset=cases()
    frozen=args.output/"cases.json"
    frozen.write_text(json.dumps(dataset,ensure_ascii=False,indent=2)+"\n")
    # Freeze and persist before loading or running the model.
    import laya
    import torch
    torch.set_num_threads(4)
    manifest=json.loads((Path(args.model)/"download.json").read_text())
    model=laya.load(args.model,device="cpu",fast=False,compile=False,
                    expected_sha256={e["path"]:e["sha256"] for e in manifest["files"]})
    runtime=VectorRuntime.load("results/catalog_retrieval/final/program.pt")
    controls=[]
    for text, expected in [("I was charged twice. Please refund the extra payment.","billing"),
                           ("The application crashes whenever I sign in.","technical"),
                           ("결제가 두 번 됐습니다. 중복 결제 금액을 환불해주세요.","billing"),
                           ("앱에 로그인하면 오류가 나고 종료됩니다.","technical")]:
        r=decision(model,text,"Which department should handle this request?",
                   [("billing","invoices, payments, refunds"),("technical","bugs, outages, system errors"),("sales","product sales"),("other","none of these")])
        controls.append({"text":text,"expected":expected,**r,"passed":r["selected"]==expected})
    records=[]
    counts=defaultdict(lambda:dict(correct=0,total=0))
    for case in dataset:
        options=[(c.name,document(runtime.registry,c.name)) for c in runtime.registry if input_types(c)==case["types"]]
        options.append(("none","None of the available functions performs the requested operation, or the request is ambiguous."))
        record={"case":case,"selection":[],"binding":[]}
        for order in range(3):
            ordered=options[order:]+options[:order]
            r=decision(model,case["intent"],"Choose the function that exactly performs the requested operation. Do not substitute a similar operation. Choose none if unsupported.",ordered)
            r["order"]=order
            r["passed"]=r["selected"]==case["expected"]
            record["selection"].append(r)
            for group in ("selection", "selection_"+case["language"],"rejection" if case["expected"]=="none" else "known_selection"):
                counts[group]["total"]+=1
                counts[group]["correct"]+=r["passed"]
        if case["expected"]!="none":
            # Correct function contract deliberately supplied to isolate binding skill.
            cap=runtime.registry.get(case["expected"])
            paths,numbers=literals(case["intent"])
            values=[n for n in numbers if 1<=n<=32] or [16]
            fields=[("width","register bit width",values,case["width"])]
            for i,(kind,role) in enumerate(zip(input_types(cap),argument_roles(cap))):
                fields.append((f"x{i}",role,paths if kind=="path" else numbers,case["operands"][i]))
            for field,role,values,expected in fields:
                options=[(str(i),str(v)) for i,v in enumerate(values)]
                for order in range(min(3,len(options))):
                    ordered=options[order:]+options[:order]
                    r=decision(model,case["intent"],f"Function: {document(runtime.registry,cap.name)} Select the user's intended {role}. Other numbers may be the register width or other operands.",ordered)
                    r.update(field=field,order=order,actual=values[int(r["selected"])],expected=expected)
                    r["passed"]=r["actual"]==expected
                    record["binding"].append(r)
                    group="binding_"+("path" if isinstance(expected,str) else "width" if field=="width" else "number")
                    counts[group]["total"]+=1
                    counts[group]["correct"]+=r["passed"]
        records.append(record)
        print(json.dumps({"case":case["name"],"selection":[r['selected'] for r in record['selection']],"binding_correct":sum(r['passed'] for r in record['binding']),"binding_total":len(record['binding'])},ensure_ascii=False),flush=True)
    all_decisions=controls+[r for c in records for k in ("selection","binding") for r in c[k]]
    summary={"protocol":"new frozen diagnostic, no tuning; three option orders are repeated measurements, not independent cases; caller supplies types; binding receives correct contract; no execution, teaching or registration",
             "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "dataset_sha256":hashlib.sha256(frozen.read_bytes()).hexdigest(),"model":manifest,
             "controls":controls,"control_correct":sum(c['passed'] for c in controls),
             "counts":dict(counts),"decisions":len(all_decisions),
             "truncated":sum(d['response'].get('usage',{}).get('truncated',False) for d in all_decisions),
             "records":records}
    (args.output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:summary[k] for k in ("control_correct","counts","decisions","truncated")}),flush=True)


if __name__=="__main__":
    main()
