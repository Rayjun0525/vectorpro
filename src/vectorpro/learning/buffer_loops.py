"""Generic indexed-buffer grammar compiled to the existing control tensors.

The grammar supplies iteration structure, like BitFold. Numeric step/body
operators and argument routing are searched over the acquired library. No task
name, Python byte transformation or stored expected file content is executed.
"""
from itertools import product

from vectorpro.host import HOST_TYPES
from vectorpro.machine import Instr


def buffer_loop_candidates(operations, input_types, max_steps, step_valid):
    paths = [f"x{i}" for i, kind in enumerate(input_types) if kind == "path"]
    values = [f"x{i}" for i, kind in enumerate(input_types) if kind == "value"] + ["zero", "one"]
    hosts = {op for op, _ in operations if op in HOST_TYPES}
    required = {"file.read", "buffer.length", "buffer.set", "file.write"}
    if not required <= hosts or max_steps < 5:
        return
    numeric = [(op, args) for op, (args, result) in operations
               if result == "value" and args and all(t == "value" for t in args) and op not in hosts]
    for source, target in product(paths, repeat=2):
        for step, signature in numeric:
            # Search unary feedback or binary feedback with a constant/input.
            for step_args in product(["index", "one"], repeat=len(signature)):
                if "index" not in step_args or not step_valid(step, step_args):
                    continue
                setup = [Instr("file.read", (source,), "buffer"),
                         Instr("buffer.length", ("buffer",), "index")]
                update = Instr(step, step_args, "index", label="body")
                bodies = [([Instr("buffer.set", ("buffer", "index", value), "last")], "fill")
                          for value in values]
                if "buffer.get" in hosts and max_steps >= 7:
                    for transform, arguments in numeric:
                        for args in product(["byte", *values], repeat=len(arguments)):
                            if "byte" not in args:
                                continue
                            bodies.append(([Instr("buffer.get", ("buffer", "index"), "byte"),
                                            Instr(transform, args, "value"),
                                            Instr("buffer.set", ("buffer", "index", "value"), "last")], "map"))
                for body, shape in bodies:
                    if len(setup) + len(body) + 2 > max_steps:
                        continue
                    # Numeric feedback is searched, including incorrect/nonhalting
                    # updates. Only case evaluation selects a working iterator.
                    body[-1] = Instr(body[-1].op, body[-1].args, body[-1].dest,
                                     test="index", then="body", otherwise="finish")
                    instrs = tuple(setup + [Instr(test="index", then="body", otherwise="finish"),
                                            update] + body + [Instr("file.write", (target, "buffer"), "result", label="finish")])
                    registers = {f"x{i}": "zero" for i in range(len(input_types))}
                    registers.update({name: "zero" for name in ("zero", "buffer", "index", "last", "byte", "value", "result")})
                    registers["one"] = "one"
                    yield instrs, registers, "result", f"buffer-{shape}"
