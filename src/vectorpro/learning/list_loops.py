"""Bounded typed body search inside an explicit reverse list-iteration grammar."""
from itertools import product
from vectorpro.machine import Instr


def list_loop_candidates(operations, input_types, max_steps, step_valid, cancelled=lambda: False):
    hosts = {name for name, _ in operations}
    if not {"list.length", "list.get"} <= hosts or max_steps < 5:
        return
    initial = {f"x{i}": kind for i, kind in enumerate(input_types)}
    sources = [(op, f"x{i}") for op, (args, result) in operations if args == ("path",) and result == "buffer"
               for i, kind in enumerate(input_types) if kind == "path"]
    sources += [(None, f"x{i}") for i, kind in enumerate(input_types) if kind == "buffer"]
    numeric = [(op, args) for op, (args, result) in operations if result == "value" and args
               and all(t == "value" for t in args) and op not in ("list.length", "buffer.length")]
    bodies = sorted(operations, key=lambda pair: {"path": 0, "buffer": 1, "value": 2}[pair[1][1]])
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
                    for body in sequences(depth):
                        output = body[-1].dest
                        # Preserve effects independently of the numeric result.
                        body = (*body[:-1], Instr(body[-1].op, body[-1].args, output,
                                                 test="index", then="body", otherwise="done"))
                        instructions = (*setup, Instr(test="index", then="body", otherwise="done"),
                                        Instr(step, step_args, "index", label="body"),
                                        Instr("list.get", (items, "index"), "item"), *body,
                                        Instr(label="done"))
                        registers = {name: "zero" for name in (*initial, "items", "index", "item", "one",
                                     *(f"b{i}" for i in range(depth)))}
                        registers["one"] = "one"
                        yield instructions, registers, output, "list-iteration"
