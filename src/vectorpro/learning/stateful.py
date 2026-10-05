"""Bounded typed vector-program search with optional guards and feedback loops."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import product
import math
from time import monotonic

from vectorpro.host import HOST_TYPES, MemoryHostContext
from vectorpro.learning.plan import LearningPlan, OutputWidth
from vectorpro.learning.buffer_loops import buffer_loop_candidates
from vectorpro.learning.list_loops import list_loop_candidates
from vectorpro.learning.registry import Capability, Registry, program_provenance
from vectorpro.machine import HALT, NEXT, Instr, assemble
from vectorpro.machine.program import MAX_ARGS


@dataclass
class StateExample:
    inputs: tuple[str | int, ...]
    before: dict[str, bytes]
    after: dict[str, bytes]
    output: int | str | None = None
    operations: tuple[str, ...] | None = None
    before_directories: tuple[str, ...] = ()
    after_directories: tuple[str, ...] = ()
    processes: tuple[dict, ...] = ()
    observations: tuple[dict, ...] = ()

    def signature(self):
        return (self.inputs, tuple(sorted(self.before.items())),
                tuple(sorted(MemoryHostContext.directory_state(self.before, self.before_directories))))

@dataclass
class StateLesson:
    input_types: tuple[str, ...]
    training: list[StateExample]
    validation: list[StateExample]
    width: int = 16
    max_steps: int = 3
    candidate_budget: int = 5000
    control_flow: bool = False
    time_budget_seconds: float = 60.0
    buffer_loops: bool = False
    execution_budget: int | None = None
    list_loops: bool = False
    allowed_operations: tuple[str, ...] | None = None
    list_reduction: bool = False
    output_type: str = "value"
    branches_only: bool = False

    def validate(self):
        if self.output_type not in ("value", "buffer"):
            raise ValueError("state output type must be value or buffer")
        if not self.input_types or any(t not in ("path", "value", "buffer") for t in self.input_types):
            raise ValueError("state inputs must declare path/value types")
        if self.width < 1 or self.max_steps < 1 or self.candidate_budget < 1:
            raise ValueError("state widths and search budgets must be positive")
        if type(self.control_flow) is not bool:
            raise ValueError("control_flow must be boolean")
        if type(self.branches_only) is not bool or (self.branches_only and not self.control_flow):
            raise ValueError("branches_only requires control_flow and a boolean flag")
        if type(self.buffer_loops) is not bool:
            raise ValueError("buffer_loops must be boolean")
        if type(self.list_loops) is not bool:
            raise ValueError("list_loops must be boolean")
        if type(self.list_reduction) is not bool or (self.list_reduction and not self.list_loops):
            raise ValueError("list_reduction requires list_loops and a boolean flag")
        if self.allowed_operations is not None and any(op not in HOST_TYPES for op in self.allowed_operations):
            raise ValueError("unknown allowed host operation")
        if self.execution_budget is not None and (type(self.execution_budget) is not int or self.execution_budget < 1):
            raise ValueError("execution budget must be a positive integer")
        if (type(self.time_budget_seconds) not in (int, float)
                or not math.isfinite(self.time_budget_seconds) or self.time_budget_seconds <= 0):
            raise ValueError("time budget must be a positive finite number")
        if not self.training or not self.validation:
            raise ValueError("state learning requires training and validation examples")
        for case in self.training + self.validation:
            from vectorpro.learning.observations import validate
            validate(case.processes, case.observations)
            if len(case.inputs) != len(self.input_types):
                raise ValueError("state input arity mismatch")
            for value, kind in zip(case.inputs, self.input_types):
                if kind == "path":
                    if not isinstance(value, str):
                        raise ValueError("path inputs must be strings")
                    MemoryHostContext.normalize(value)
                elif kind == "buffer":
                    if not isinstance(value, str) or len(value) % 2 or any(c not in "0123456789abcdefABCDEF" for c in value):
                        raise ValueError("buffer state inputs require hexadecimal byte pairs")
                elif type(value) is not int or not 0 <= value < 1 << self.width:
                    raise ValueError("numeric state inputs must fit register width")
            for files in (case.before, case.after):
                for name, data in files.items():
                    if MemoryHostContext.normalize(name) != name or not isinstance(data, bytes):
                        raise ValueError("state files require normalized paths and bytes")
            for files, directories in ((case.before, case.before_directories), (case.after, case.after_directories)):
                if (not isinstance(directories, (list, tuple)) or len(set(directories)) != len(directories)
                        or any(not isinstance(p, str) or MemoryHostContext.normalize(p) != p for p in directories)):
                    raise ValueError("state directories require distinct normalized paths")
                MemoryHostContext.directory_state(files, directories)
            if case.output is not None:
                if self.output_type == "buffer":
                    if not isinstance(case.output, str) or len(case.output) % 2 or any(c not in "0123456789abcdefABCDEF" for c in case.output):
                        raise ValueError("buffer output requires hexadecimal byte pairs")
                elif type(case.output) is not int or not 0 <= case.output < 1 << self.width:
                    raise ValueError("state output must fit register width")
            if case.operations is not None and any(op not in HOST_TYPES for op in case.operations):
                raise ValueError("operation traces must contain host operation names")
        train = [c.signature() for c in self.training]
        validation = [c.signature() for c in self.validation]
        if len(set(train)) != len(train) or len(set(validation)) != len(validation) or set(train) & set(validation):
            raise ValueError("state training/validation initial conditions must be distinct")

    @classmethod
    def from_dict(cls, data):
        def cases(name):
            return [StateExample(tuple(c["inputs"]),
                                 {p: bytes.fromhex(v) for p, v in c["before"].items()},
                                 {p: bytes.fromhex(v) for p, v in c["after"].items()}, c.get("output"),
                                 tuple(c["operations"]) if "operations" in c else None,
                                 c.get("before_directories", ()), c.get("after_directories", ()), tuple(c.get("processes", ())), tuple(c.get("observations", ())))
                    for c in data[name]]
        return cls(tuple(data["input_types"]), cases("training"), cases("validation"),
                   data.get("width", 16), data.get("max_steps", 3), data.get("candidate_budget", 5000),
                   data.get("control_flow", False), data.get("time_budget_seconds", 60.0),
                   data.get("buffer_loops", False), data.get("execution_budget"), data.get("list_loops", False),
                   None, data.get("list_reduction", False), data.get("output_type", "value"), data.get("branches_only", False))


@dataclass
class StateOutcome:
    capability: Capability | None
    history: list[dict] = field(default_factory=list)


def evaluate(executable, case: StateExample, lesson: StateLesson) -> bool:
    context = MemoryHostContext(case.before, directories=case.before_directories)
    context.processes = case.processes
    context.observations = case.observations
    inputs = tuple(context.put(value.encode("utf-8")) if kind == "path" else context.put(bytes.fromhex(value)) if kind == "buffer" else value
                   for value, kind in zip(case.inputs, lesson.input_types))
    try:
        with context.activate():
            output = executable([inputs], lesson.width)[0]
        return (context.files == case.after
                and getattr(context, "process_index", 0) == len(case.processes)
                and getattr(context, "observation_index", 0) == len(case.observations)
                and context.directories == MemoryHostContext.directory_state(case.after, case.after_directories)
                and (case.output is None or (bytes(context.buffers[output]) == bytes.fromhex(case.output)
                     if lesson.output_type == "buffer" else output == case.output))
                and (case.operations is None or tuple(e["operation"] for e in context.events) == case.operations))
    except (OSError, ValueError, KeyError, IndexError, RuntimeError):
        return False


def prefilter_instructions(registry, instructions, registers, output, case, lesson, cancelled):
    """Generic scalar simulation rejects bad search candidates; tensors recheck survivors.

    No task-name dispatch. Calls, writes and branch routing follow the candidate.
    Native host APIs are never used. This is not the adopted execution backend.
    """
    context = MemoryHostContext(case.before, directories=case.before_directories)
    context.processes = case.processes
    context.observations = case.observations
    values = {name: int(value == "one") for name, value in registers.items()}
    for i, (value, kind) in enumerate(zip(case.inputs, lesson.input_types)):
        values[f"x{i}"] = context.put(value.encode()) if kind == "path" else context.put(bytes.fromhex(value)) if kind == "buffer" else value
    labels = {ins.label: i for i, ins in enumerate(instructions) if ins.label}
    pc = 0
    mask = (1 << lesson.width) - 1
    try:
        with context.activate():
            for _ in range(lesson.execution_budget or 20000):
                if cancelled():
                    return False
                if pc == len(instructions):
                    return (context.files == case.after
                            and getattr(context, "process_index", 0) == len(case.processes)
                            and getattr(context, "observation_index", 0) == len(case.observations)
                            and context.directories == MemoryHostContext.directory_state(case.after, case.after_directories)
                            and (case.output is None or (bytes(context.buffers[values[output]]) == bytes.fromhex(case.output)
                                 if lesson.output_type == "buffer" else values[output] == case.output))
                            and (case.operations is None or tuple(e["operation"] for e in context.events) == case.operations))
                ins = instructions[pc]
                if ins.op:
                    cap = registry.get(ins.op)
                    args = tuple(values[a] for a in ins.args)
                    result = (context.invoke(cap.provenance["operation"], args, lesson.width)
                              if cap.provenance["kind"] == "host" else cap.run([args], lesson.width)[0])
                    if ins.dest is not None:
                        values[ins.dest] = result & mask
                target = ins.then if ins.test is None or values[ins.test] else ins.otherwise
                pc = pc + 1 if target == NEXT else len(instructions) if target == HALT else labels[target]
    except (OSError, ValueError, KeyError, IndexError, RuntimeError):
        return False
    return False


def control_candidates(instructions, input_types, result_types, enabled, allow_loops=True):
    """One structured guard or while region; no operation/task-specific recipes.

    A while region feeds a numeric call's result back into one of its numeric
    operands. A pre-test also handles zero iterations. Other results retain
    their register routes. Constants and handles are never feedback targets.
    """
    yield instructions, instructions[-1].dest, "sequence"
    if not enabled:
        return
    size = len(instructions)
    available = {f"x{i}": kind for i, kind in enumerate(input_types)}
    for start in range(size):
        for test, kind in available.items():
            if kind != "value":
                continue
            for end in range(start, size):
                for nonzero in (True, False):
                    labeled = [replace(ins, label=f"s{i}") for i, ins in enumerate(instructions)]
                    exit_label = f"s{end + 1}" if end + 1 < size else HALT
                    guard = Instr(test=test, then=f"s{start}" if nonzero else exit_label,
                                  otherwise=exit_label if nonzero else f"s{start}")
                    labeled.insert(start, guard)
                    yield tuple(labeled), instructions[-1].dest, "branch"
            if not allow_loops:
                continue
            for update in range(start, size):
                if result_types[update] != "value" or test not in instructions[update].args:
                    continue
                old = instructions[update].dest
                for end in range(update, size):
                    labeled = []
                    for i, ins in enumerate(instructions):
                        # Only downstream references see the updated value.
                        args = tuple(test if a == old and i > update else a for a in ins.args)
                        labeled.append(replace(ins, args=args, dest=test if i == update else ins.dest,
                                               label=f"s{i}"))
                    exit_label = f"s{end + 1}" if end + 1 < size else HALT
                    labeled[end] = replace(labeled[end], test=test, then=f"s{start}", otherwise=exit_label)
                    labeled.insert(start, Instr(test=test, then=f"s{start}", otherwise=exit_label))
                    # The feedback value is also a defined result for zero iterations.
                    yield tuple(labeled), test, "while"
                    if instructions[-1].dest != old:
                        yield tuple(labeled), instructions[-1].dest, "while"
        available[instructions[start].dest] = result_types[start]


def learn_stateful(registry: Registry, name: str, lesson: StateLesson) -> StateOutcome:
    lesson.validate()
    if name in registry:
        raise ValueError("cannot replace an existing state capability")
    deadline = monotonic() + lesson.time_budget_seconds

    def timed_out():
        return monotonic() >= deadline

    def matches(executable, cases):
        for case in cases:
            if timed_out() or not evaluate(executable, case, lesson):
                return False
        return not timed_out()
    # Numeric capabilities are called as learned, never replaced by Python arithmetic.
    # Typed state procedures can also be reused as one call (their internal steps
    # still execute normally). Path/handle registers are not numeric arguments.
    operations = []
    for cap in registry:
        if cap.plan.arity > MAX_ARGS:
            continue
        provenance = cap.provenance
        if lesson.allowed_operations is not None and cap.executable.effects:
            from vectorpro.contracts import _closure, _operations
            if set(_operations(_closure(registry, cap.name))) - set(lesson.allowed_operations):
                continue
        if provenance.get("kind") == "host":
            signature = HOST_TYPES[provenance["operation"]]
        elif "input_types" in provenance and "output_type" in provenance:
            signature = (tuple(provenance["input_types"]), provenance["output_type"])
        elif not cap.executable.effects:
            signature = (("value",) * cap.plan.arity, "value")
        else:
            continue  # Old effectful programs without a type contract cannot be inferred safely.
        operations.append((cap.name, signature))
    # Explore data producers before terminal scalar/effect calls; keep all options.
    if any(case.processes or case.observations for case in lesson.training):
        operations.sort(key=lambda op: {"buffer": 0, "path": 1, "value": 2}[op[1][1]])
    plan = LearningPlan(name, "learned state transformation", len(lesson.input_types), OutputWidth.SAME)
    initial = {f"x{i}": "zero" for i in range(plan.arity)} | {"zero": "zero", "one": "one"}
    types = {f"x{i}": t for i, t in enumerate(lesson.input_types)} | {"zero": "value", "one": "value"}
    frontier = [((), types, ())]
    tried = 0
    if lesson.buffer_loops or lesson.list_loops:
        if lesson.output_type != "value":
            raise ValueError("loop search currently requires value outputs")
        def step_valid(op, args):
            if timed_out() or registry.get(op).executable.effects:
                return False
            probes = sorted({1, 2, 3, min(17, (1 << lesson.width) - 1)})
            probes = [n for n in probes if n < 1 << lesson.width]
            rows = [tuple(n if arg == "index" else 1 for arg in args) for n in probes]
            try:
                return registry.get(op).run(rows, lesson.width) == [n - 1 for n in probes]
            except (ValueError, RuntimeError):
                return False

        from itertools import chain
        candidates = chain(buffer_loop_candidates(operations, lesson.input_types, lesson.max_steps, step_valid) if lesson.buffer_loops else (),
                           list_loop_candidates(operations, lesson.input_types, lesson.max_steps, step_valid, timed_out,
                                                lesson.control_flow, lesson.list_reduction) if lesson.list_loops else ())
        for instructions, registers, output, control in candidates:
            if timed_out() or tried >= lesson.candidate_budget:
                reason = "time budget" if timed_out() else "candidate budget"
                return StateOutcome(None, [{"candidates": tried, "accepted": False, "reason": reason}])
            tried += 1
            if control.startswith("list-") and not all(prefilter_instructions(registry, instructions, registers, output,
                    case, lesson, timed_out) for case in lesson.training):
                continue
            program = assemble(instructions, [f"x{i}" for i in range(plan.arity)], registers, output, registry.key_of)
            provenance = program_provenance(program, "state example search", input_types=list(lesson.input_types),
                                            output_type="value", teaching_width=lesson.width, control=control,
                                            execution_budget=lesson.execution_budget)
            executable = registry.build(plan, provenance)
            if matches(executable, lesson.training) and matches(executable, lesson.validation):
                history = [{"strategy": "state search", "steps": sum(i.op is not None for i in instructions),
                            "tensor_steps": len(instructions), "candidates": tried, "control": control,
                            "training_cases": len(lesson.training), "validation_cases": len(lesson.validation), "accepted": True}]
                capability = Capability(plan, executable, provenance, history)
                registry.add(capability)
                return StateOutcome(capability, history)
    for depth in range(1, lesson.max_steps + 1):
        next_frontier = []
        for prefix, available, prefix_types in frontier:
            for op, (arguments, result_type) in operations:
                choices = [[r for r, t in available.items() if t == kind] for kind in arguments]
                for args in product(*choices):
                    destination = f"t{depth}"
                    instructions = prefix + (Instr(op, tuple(args), destination),)
                    registers = initial | {f"t{i}": "zero" for i in range(1, depth + 1)}
                    result_types = prefix_types + (result_type,)
                    for candidate, output, control in control_candidates(
                            instructions, lesson.input_types, result_types, lesson.control_flow,
                            allow_loops=not lesson.branches_only):
                        if (available | {destination: result_type})[output] != lesson.output_type:
                            continue
                        if timed_out():
                            return StateOutcome(None, [{"candidates": tried, "accepted": False, "reason": "time budget"}])
                        if tried >= lesson.candidate_budget:
                            return StateOutcome(None, [{"candidates": tried, "accepted": False, "reason": "candidate budget"}])
                        tried += 1
                        if (control == "sequence" and any(case.processes or case.observations for case in lesson.training)
                                and not all(prefilter_instructions(registry, candidate, registers, output,
                                            case, lesson, timed_out) for case in lesson.training)):
                            continue
                        program = assemble(candidate, [f"x{i}" for i in range(plan.arity)],
                                           registers, output, registry.key_of)
                        provenance = program_provenance(program, "state example search",
                                                        input_types=list(lesson.input_types),
                                                        output_type=(available | {destination: result_type})[output],
                                                        teaching_width=lesson.width, control=control,
                                                        execution_budget=lesson.execution_budget,
                                                        **({"branch_description_version": 2} if lesson.branches_only else {}))
                        executable = registry.build(plan, provenance)
                        if matches(executable, lesson.training):
                            if matches(executable, lesson.validation):
                                history = [{"strategy": "state search", "steps": depth, "candidates": tried,
                                            "control": control, "tensor_steps": len(candidate),
                                            "training_cases": len(lesson.training), "validation_cases": len(lesson.validation),
                                            "accepted": True}]
                                capability = Capability(plan, executable, provenance, history)
                                registry.add(capability)
                                return StateOutcome(capability, history)
                    next_frontier.append((instructions, available | {destination: result_type}, result_types))
        frontier = next_frontier
    reason = "time budget" if timed_out() else "step budget"
    return StateOutcome(None, [{"candidates": tried, "accepted": False, "reason": reason}])
