"""Caller-owned finite state predicates; no natural-language interpretation."""
import copy


def validate_goal(goal):
    if not isinstance(goal, dict) or set(goal) != {"intent", "rules"}:
        raise ValueError("goal requires exact intent and rules")
    if not isinstance(goal["intent"], str) or not 1 <= len(goal["intent"]) <= 2000:
        raise ValueError("goal needs bounded intent")
    rules = goal["rules"]
    if not isinstance(rules, list) or not 1 <= len(rules) <= 32:
        raise ValueError("goal requires 1..32 rules")
    for rule in rules:
        if not isinstance(rule, dict):
            raise ValueError("invalid goal rule")
        kind = rule.get("kind")
        fields = {"kind", "parameters"} if kind == "unchanged_except" else {"kind", "parameter", "source"} if kind == "equals_initial" else {"kind", "parameter"}
        if kind not in ("absent", "present", "unchanged", "equals_initial", "unchanged_except") or set(rule) != fields:
            raise ValueError("unsupported goal rule")
        names = rule["parameters"] if kind == "unchanged_except" else [rule["parameter"]] + ([rule["source"]] if kind == "equals_initial" else [])
        if not isinstance(names, list) or not names or len(names) > 16 or any(not isinstance(n, str) or not 1 <= len(n) <= 100 for n in names):
            raise ValueError("goal references bounded parameter names")
    return copy.deepcopy(goal)


def check_goal(goal, record):
    """All supplied training, validation and hidden observations must satisfy rules."""
    if record["intent"] != goal["intent"] or "state_lesson" not in record:
        raise ValueError("goal requires matching state evidence")
    names = [p["name"] for p in record["interface"]["parameters"]]
    lesson = record["state_lesson"]
    cases = lesson["training"] + lesson["validation"] + record["heldout"]
    for case in cases:
        paths = dict(zip(names, case["inputs"]))
        before, after = case["before"], case["after"]
        before_dirs, after_dirs = set(case.get("before_directories", [])), set(case.get("after_directories", []))
        for rule in goal["rules"]:
            kind = rule["kind"]
            try:
                if kind == "unchanged_except":
                    exceptions = {paths[n] for n in rule["parameters"]}
                    valid = ({k: v for k, v in before.items() if k not in exceptions} == {k: v for k, v in after.items() if k not in exceptions}
                             and before_dirs - exceptions == after_dirs - exceptions)
                else:
                    path = paths[rule["parameter"]]
                    if kind == "absent":
                        valid = path not in after and path not in after_dirs
                    elif kind == "present":
                        valid = path in after or path in after_dirs
                    elif kind == "unchanged":
                        valid = (path in before or path in before_dirs) and before.get(path) == after.get(path) and (path in before_dirs) == (path in after_dirs)
                    else:
                        source = paths[rule["source"]]
                        valid = source in before and path in after and after[path] == before[source]
            except KeyError as error:
                raise ValueError("goal references unknown interface parameter") from error
            if not valid:
                raise ValueError("reference observations contradict caller goal")
    return len(cases)
