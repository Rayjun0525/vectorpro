"""Bounded typed body search inside an explicit reverse list-iteration grammar."""
from itertools import product
from vectorpro.machine import Instr


def list_loop_candidates(operations, input_types, max_steps, step_valid, cancelled=lambda: False,
                         control_flow=False, reduction=False):
    hosts = {name for name, _ in operations}
    if not {"list.length", "list.get"} <= hosts or max_steps < 5:
        return
    initial = {f"x{i}": kind for i, kind in enumerate(input_types)}
    sources = [(op, f"x{i}") for op, (args, result) in operations if args == ("path",) and result == "buffer"
               for i, kind in enumerate(input_types) if kind == "path"]
    sources += [(None, f"x{i}") for i, kind in enumerate(input_types) if kind == "buffer"]
    # A list producer precedes arbitrary byte buffers in the list grammar.
    # Other sources remain eligible within the same bounded search.
    sources.sort(key=lambda source: 0 if source[0] == "directory.list" else 1)
    numeric = [(op, args) for op, (args, result) in operations if result == "value" and args
               and all(t == "value" for t in args) and op not in ("list.length", "buffer.length")]
    bodies = sorted(operations, key=lambda pair: {"path": 0, "buffer": 1, "value": 2}[pair[1][1]])
    available = initial | {"item": "buffer"}
    predicates = [None]
    if control_flow:
        for op, (args, result) in operations:
            if result != "value":
                continue
            choices = [[r for r, kind in available.items() if kind == a] for a in args]
            for route in product(*choices):
                if "item" in route:
                    predicates.extend((op, route, polarity) for polarity in (True, False))
        # Try comparisons between an item and an actual caller parameter first.
        # This changes search order only; no predicates are discarded.
        predicates.sort(key=lambda p: (0 if p and "item" in p[1] and any(a in initial for a in p[1]) else 1,
                                       0 if p and p[1][0] == "item" else 1))
    reducers = [(op, args) for op, signature in numeric if signature == ("value", "value")
                for args in (("acc", "value"), ("value", "acc"))] if reduction else [None]
    for source_op, source in sources:
        if cancelled():
            return
        for step, signature in numeric:
            for step_args in product(("index", "one"), repeat=len(signature)):
                if "index" not in step_args or not step_valid(step, step_args):
                    continue
                setup = ([Instr(source_op, (source,), "items")] if source_op else [])
                items = "items" if source_op else source
                setup += [Instr("list.length", (items,), "index")]
                limit = max_steps - len(setup) - 2  # numeric update and element extraction
                def sequences(depth, prefix=(), available=None):
                    available = available or (initial | {"item": "buffer"})
                    for op, (arguments, result) in bodies:
                        if cancelled():
                            return
                        # This grammar searches one terminal value/effect call with
                        # a typed argument-expression DAG, not arbitrary body blocks.
                        if depth > 1 and result == "value":
                            continue
                        choices = [[r for r, kind in available.items() if kind == a] for a in arguments]
                        for args in product(*choices):
                            if cancelled():
                                return
                            destination = f"b{len(prefix)}"
                            body = prefix + (Instr(op, tuple(args), destination),)
                            if depth == 1:
                                if result == "value":
                                    needed = set(body[-1].args)
                                    for prior in reversed(body[:-1]):
                                        if prior.dest in needed:
                                            needed.update(prior.args)
                                    if all(prior.dest in needed for prior in body[:-1]) and "item" in needed:
                                        yield body
                            else:
                                yield from sequences(depth - 1, body, available | {destination: result})
                for depth in range(1, limit + 1):
                    for predicate, body in ((p, b) for p in predicates for b in sequences(depth)):
                        value = body[-1].dest
                        original = body
                        for reducer in reducers:
                            if cancelled():
                                return
                            if len(setup) + 2 + len(original) + bool(predicate) + bool(reducer) > max_steps:
                                continue
                            output = "acc" if reducer else value
                            if predicate or reducer:
                                guard = [] if predicate is None else [Instr(predicate[0], predicate[1], "predicate"),
                                    Instr(test="predicate", then="action" if predicate[2] else "latch",
                                          otherwise="latch" if predicate[2] else "action")]
                                action = (Instr(original[0].op, original[0].args, original[0].dest, label="action"), *original[1:])
                                reduce_call = [] if reducer is None else [Instr(reducer[0], tuple(value if a == "value" else a for a in reducer[1]), "acc")]
                                instructions = (*setup, Instr(test="index", then="body", otherwise="done"),
                                    Instr(step, step_args, "index", label="body"), Instr("list.get", (items, "index"), "item"),
                                    *guard, *action, *reduce_call,
                                    Instr(test="index", then="body", otherwise="done", label="latch"), Instr(label="done"))
                            else:
                                body = (*original[:-1], Instr(original[-1].op, original[-1].args, value,
                                                 test="index", then="body", otherwise="done"))
                                instructions = (*setup, Instr(test="index", then="body", otherwise="done"),
                                        Instr(step, step_args, "index", label="body"),
                                        Instr("list.get", (items, "index"), "item"), *body,
                                        Instr(label="done"))
                            registers = {name: "zero" for name in (*initial, "items", "index", "item", "one", "acc", "predicate",
                                     *(f"b{i}" for i in range(depth)))}
                            registers["one"] = "one"
                            shape = "list-reduction" if reduction else "list-selection" if predicate else "list-iteration"
                            yield instructions, registers, output, shape
