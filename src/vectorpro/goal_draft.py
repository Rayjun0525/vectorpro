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


def propose_goal(model, providers, intent, *, encoding="rules", goal_memory=None):
    if encoding not in ("per_parameter", "states", "rules", "compact"):
        raise ValueError("goal encoding must be compact, per_parameter, states or rules")
    if not isinstance(intent, str) or not 1 <= len(intent) <= 2000:
        raise ValueError("goal draft requires bounded intent")
    interfaces = [p["interface"]["parameters"] for p in providers.describe()]
    if not interfaces or any(p != interfaces[0] for p in interfaces) or any(p["type"] != "path" for p in interfaces[0]):
        return {"status": "needs_input", "question": "목표를 표현할 공통 파일 경로 인터페이스가 필요합니다."}
    names = [p["name"] for p in interfaces[0]]
    if goal_memory is not None:
        draft = goal_memory.propose(intent)
        if draft["status"] == "needs_goal_review":
            for rule in draft["goal"]["rules"]:
                referenced = rule["parameters"] if rule["kind"] == "unchanged_except" else [rule["parameter"]] + ([rule["source"]] if "source" in rule else [])
                if any(n not in names for n in referenced):
                    return {"status": "goal_draft_failed", "reason": "goal memory interface mismatch"}
        return draft
    if encoding == "compact":
        from vectorpro.goal_interpretation import propose_interpretation
        return propose_interpretation(model, interfaces[0], intent)
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
    if encoding == "per_parameter":
        states = {}
        allowed = ["unchanged", "absent", "present"] + ["initial:" + n for n in names] + ["unsupported", "unclear"]
        single_tool = tool("describe_parameter", "Choose the requested parameter's final state; no execution",
            {"state": {"type": "string", "enum": allowed}}, ("state",))
        for parameter_data in interfaces[0]:
            name = parameter_data["name"]
            messages = [{"role": "system", "content":
                "Read the actual request and choose ONE final state for the specified parameter. "
                "unchanged: keep this file exactly. absent: this file must be gone. "
                "present: require existence only. initial:NAME: exact original bytes from NAME. "
                "unsupported: ANY requested encryption, byte transformation, computation, process or network effect. "
                "unclear: the intended final effect is unspecified. "
                "Reject the WHOLE request if any effect is unsupported, even if this parameter is only preserved. "
                "Use unchanged for preservation, not initial:self. This is a draft requiring caller review.\n"
                "Interface: " + json.dumps(interfaces[0]) + "\nParameter to describe: " + json.dumps(parameter_data)}]
            if len(names) >= 2:
                first, second = names[:2]
                examples = [
                    (f"Duplicate {first} into {second}, keeping {first} unchanged.", "initial:" + first if name == second else "unchanged"),
                    (f"Move {first} into {second}; {first} must no longer exist.", "absent" if name == first else "initial:" + first if name == second else "unchanged"),
                    (f"Keep {first} and remove {second}.", "absent" if name == second else "unchanged"),
                    ("Keep all files exactly as they are.", "unchanged"),
                    ("Encrypt a file using a key.", "unsupported"),
                    ("Reverse the order of bytes.", "unsupported"),
                    ("Do something useful with the files.", "unclear")]
                for request, answer in examples:
                    messages.extend([{"role": "user", "content": request},
                        {"role": "assistant", "tool_calls": [{"id": "notation", "type": "function",
                         "function": {"name": "describe_parameter", "arguments": json.dumps({"state": answer})}}]}])
            messages.append({"role": "user", "content": intent})
            message = model.complete(messages, [single_tool])
            try:
                calls = message.get("tool_calls", [])
                if message.get("role") != "assistant" or len(calls) != 1 or calls[0]["function"]["name"] != "describe_parameter":
                    raise ValueError("one parameter state required")
                argument = calls[0]["function"]["arguments"]
                argument = json.loads(argument) if isinstance(argument, str) else argument
                if set(argument) != {"state"} or argument["state"] not in allowed:
                    raise ValueError("invalid parameter state")
                if argument["state"] in ("unsupported", "unclear"):
                    return {"status": "needs_input", "question": "요청 전체를 표현할 수 있는 명확한 파일 상태 목표가 필요합니다.", "model_assessment": argument["state"]}
                states[name] = argument["state"]
            except (ValueError, KeyError, TypeError) as error:
                return {"status": "goal_draft_failed", "reason": str(error)}
        converted = [{"kind": "equals_initial", "parameter": n, "source": v[8:]} if v.startswith("initial:") else {"kind": v, "parameter": n} for n, v in states.items()]
        converted.append({"kind": "unchanged_except", "parameters": [n for n, v in states.items() if v != "unchanged"]})
        goal = validate_goal({"intent": intent, "rules": converted})
        return {"status": "needs_goal_review", "goal": goal, "proposal_sha256": digest(goal),
                "review": review_lines(goal), "intent_independently_verified": False}
    if encoding == "states":
        support_tool = tool("assess_goal", "Classify whether the entire requested effect can be expressed by file-state predicates",
            {"decision": {"type": "string", "enum": ["supported", "unsupported", "ambiguous"]}}, ("decision",))
        support_messages = [{"role": "system", "content":
            "Classify the request only. supported: exact copying, moving, deletion, existence, or preserving file bytes. "
            "unsupported: encrypting, changing byte values/order, calculating, processes or networking. "
            "ambiguous: intended final effect is not stated. Classify the ENTIRE request; any unsupported effect makes it unsupported. "
            "This is model triage, not independent verification."}]
        for example, decision in (("Duplicate a file while keeping its original.", "supported"),
            ("Rename a file and remove its old name.", "supported"),
            ("Encrypt a file with a key.", "unsupported"),
            ("Reverse all the bytes in a file.", "unsupported"),
            ("Do something with these files.", "ambiguous")):
            support_messages.extend([{"role": "user", "content": example},
                {"role": "assistant", "tool_calls": [{"id": "support-example", "type": "function",
                 "function": {"name": "assess_goal", "arguments": json.dumps({"decision": decision})}}]}])
        support_messages.append({"role": "user", "content": intent})
        assessment = model.complete(support_messages, [support_tool])
        try:
            calls = assessment.get("tool_calls", [])
            if assessment.get("role") != "assistant" or len(calls) != 1 or calls[0]["function"]["name"] != "assess_goal":
                raise ValueError("one support assessment required")
            argument = calls[0]["function"]["arguments"]
            argument = json.loads(argument) if isinstance(argument, str) else argument
            if set(argument) != {"decision"} or argument["decision"] not in ("supported", "unsupported", "ambiguous"):
                raise ValueError("invalid support decision")
            if argument["decision"] != "supported":
                return {"status": "needs_input", "question": "요청 전체를 표현할 수 있는 명확한 파일 상태 목표가 필요합니다.", "model_assessment": argument["decision"]}
        except (ValueError, KeyError, TypeError) as error:
            return {"status": "goal_draft_failed", "reason": str(error)}
    tools = [tool("propose_goal", "Draft final-state predicates; this does not authorize execution",
        {"rules": {"type": "array", "items": {"anyOf": rules}, "minItems": 1, "maxItems": 32}}, ("rules",)),
        tool("ask_user", "Ask for missing or unsupported goal semantics", {"question": {"type": "string"}}, ("question",))]
    if encoding == "states":
        state = {"type": "string", "enum": ["unchanged", "absent", "present"] + ["initial:" + n for n in names]}
        tools[0] = tool("propose_goal", "Describe each parameter's final state without executing",
            {"states": {"type": "object", "properties": {n: state for n in names},
                        "required": names, "additionalProperties": False}}, ("states",))
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
    if encoding == "states":
        messages[0]["content"] = (
            "Describe the FINAL STATE of EACH file parameter. Make one tool call.\n"
            "unchanged = keep this existing file and its bytes.\n"
            "absent = this file must NOT exist afterward.\n"
            "present = require existence only, with no byte condition.\n"
            "initial:NAME = final bytes must equal the ORIGINAL bytes of parameter NAME.\n"
            "Use initial:NAME for exact byte duplication, not for encryption or transformation.\n"
            "Read explicit keep/remove conditions carefully. A request saying a file must not exist means absent.\n"
            "Return states for ALL parameters. Other files and folders will be preserved by the backend.\n"
            "Use ask_user only for ambiguous or unsupported effects: encryption, content transformations, processes, network, numeric results.\n"
            "Use unchanged rather than initial:NAME when retaining that same NAME unchanged. "
            "A trusted caller reviews this draft before any execution.\nInterface: " + json.dumps(interfaces[0]))
        # Demonstrate the notation, not a task recipe or the evaluation answers.
        examples = []
        if len(names) >= 2:
            first, second = names[:2]
            base = {n: "unchanged" for n in names}
            for request, states in (
                (f"Keep {first} as it is; {second} should contain the original bytes of {first}.", {**base, second: "initial:" + first}),
                (f"{first} must be gone; {second} must contain the original bytes of {first}.", {**base, first: "absent", second: "initial:" + first}),
                (f"Keep {first}; remove {second}.", {**base, second: "absent"})):
                examples.extend([{"role": "user", "content": request},
                    {"role": "assistant", "tool_calls": [{"id": "example", "type": "function",
                     "function": {"name": "propose_goal", "arguments": json.dumps({"states": states})}}]}])
        examples.extend([{"role": "user", "content": "Encrypt a file with a secret key."},
            {"role": "assistant", "tool_calls": [{"id": "example", "type": "function",
             "function": {"name": "ask_user", "arguments": json.dumps({"question": "Encryption is outside these state predicates; provide a supported goal."})}}]}])
        messages = [messages[0], *examples, {"role": "user", "content": intent}]
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
        if function["name"] != "propose_goal" or set(args) != ({"states"} if encoding == "states" else {"rules"}):
            raise ValueError("only draft tools are available")
        if encoding == "states":
            states = args["states"]
            if not isinstance(states, dict) or set(states) != set(names):
                raise ValueError("states must cover every interface parameter exactly")
            converted, changed = [], []
            for name in names:
                value = states[name]
                if value not in state["enum"]:
                    raise ValueError("unknown final state")
                if value.startswith("initial:"):
                    converted.append({"kind": "equals_initial", "parameter": name, "source": value[8:]})
                else:
                    converted.append({"kind": value, "parameter": name})
                if value != "unchanged":
                    changed.append(name)
            converted.append({"kind": "unchanged_except", "parameters": changed})
            args = {"rules": converted}
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
