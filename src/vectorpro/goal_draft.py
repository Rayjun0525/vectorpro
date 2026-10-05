"""One-shot natural-language goal drafts; only a trusted caller can accept one."""
import json
from vectorpro.acquisition import digest
from vectorpro.goal_evidence import validate_goal


def review_lines(goal):
    lines = []
    for rule in goal["rules"]:
        kind = rule["kind"]
        if kind == "unchanged_except":
            lines.append("허용된 변경 항목: " + ", ".join(rule["parameters"]) + "; 그 밖의 파일과 폴더는 모두 보존")
        elif kind == "equals_initial":
            lines.append(f"{rule['parameter']} 파일 내용 = 작업 전 {rule['source']} 파일 내용")
        else:
            labels = {"absent": "파일과 폴더 모두 없어야 함", "present": "파일 또는 폴더가 존재해야 함", "unchanged": "기존 항목의 내용과 종류를 보존해야 함"}
            lines.append(f"{rule['parameter']}: {labels[kind]}")
    return lines


def propose_goal(model, providers, intent):
    if not isinstance(intent, str) or not 1 <= len(intent) <= 2000:
        raise ValueError("goal draft requires bounded intent")
    interfaces = [p["interface"]["parameters"] for p in providers.describe()]
    if not interfaces or any(p != interfaces[0] for p in interfaces) or any(p["type"] != "path" for p in interfaces[0]):
        return {"status": "needs_input", "question": "목표를 표현할 공통 파일 경로 인터페이스가 필요합니다."}
    names = [p["name"] for p in interfaces[0]]
    parameter = {"type": "string", "enum": names}
    rules = []
    for kind in ("absent", "present", "unchanged", "equals_initial", "unchanged_except"):
        properties = {"kind": {"const": kind}}
        if kind == "unchanged_except":
            properties["parameters"] = {"type": "array", "items": parameter, "minItems": 1, "maxItems": len(names), "uniqueItems": True}
        else:
            properties["parameter"] = parameter
            if kind == "equals_initial":
                properties["source"] = parameter
        rules.append({"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False})
    from vectorpro.agent import tool
    tools = [tool("propose_goal", "Draft final-state predicates; this does not authorize execution",
        {"rules": {"type": "array", "items": {"anyOf": rules}, "minItems": 1, "maxItems": 32}}, ("rules",)),
        tool("ask_user", "Ask for missing or unsupported goal semantics", {"question": {"type": "string"}}, ("question",))]
    messages = [{"role": "system", "content":
        "Translate the request into FINAL STATE conditions, not actions or a reference choice. "
        "absent: no file or directory remains. present: file or directory exists. "
        "unchanged: existing contents and item kind preserved. "
        "equals_initial: final parameter file bytes equal initial source file bytes. "
        "unchanged_except: every other file and directory stays identical. "
        "Include explicit source removal or source preservation requested, and a frame condition. "
        "These conditions cannot express encryption, byte transformations, processes, network, or numeric results. "
        "Use ask_user for unsupported or ambiguous goals. Do not approximate an unsupported goal with copying. "
        "Make exactly one tool call. A trusted caller must review the draft separately.\nInterface: " + json.dumps(interfaces[0])},
        {"role": "user", "content": intent}]
    message = model.complete(messages, tools)
    try:
        calls = message.get("tool_calls", [])
        if message.get("role") != "assistant" or len(calls) != 1:
            raise ValueError("goal drafting requires exactly one tool call")
        function = calls[0]["function"]
        args = json.loads(function["arguments"]) if isinstance(function["arguments"], str) else function["arguments"]
        if function["name"] == "ask_user":
            if set(args) != {"question"} or not isinstance(args["question"], str) or not 1 <= len(args["question"]) <= 2000:
                raise ValueError("invalid goal question")
            return {"status": "needs_input", "question": args["question"]}
        if function["name"] != "propose_goal" or set(args) != {"rules"}:
            raise ValueError("only draft tools are available")
        goal = validate_goal({"intent": intent, "rules": args["rules"]})
        for rule in goal["rules"]:
            referenced = rule["parameters"] if rule["kind"] == "unchanged_except" else [rule["parameter"]] + ([rule["source"]] if "source" in rule else [])
            if any(n not in names for n in referenced):
                raise ValueError("unknown goal parameter")
        if sum(r["kind"] == "unchanged_except" for r in goal["rules"]) != 1 or len(goal["rules"]) < 2:
            raise ValueError("draft needs state conditions and one frame condition")
        return {"status": "needs_goal_review", "goal": goal, "proposal_sha256": digest(goal),
                "review": review_lines(goal), "intent_independently_verified": False}
    except (ValueError, KeyError, TypeError) as error:
        return {"status": "goal_draft_failed", "reason": str(error)}


def accept_goal(proposal, approved_sha256):
    """Caller acknowledges the exact draft; never expose this function to a model."""
    if not isinstance(proposal, dict) or proposal.get("status") != "needs_goal_review":
        raise ValueError("a reviewable draft is required")
    goal = validate_goal(proposal["goal"])
    if not isinstance(approved_sha256, str) or approved_sha256 != digest(goal) or proposal.get("proposal_sha256") != approved_sha256:
        raise ValueError("caller approval must match the exact goal digest")
    if proposal.get("review") != review_lines(goal):
        raise ValueError("review text does not match the goal")
    return goal
