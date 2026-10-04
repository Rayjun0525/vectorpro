"""Task-agnostic bounded search for typed, effectful vector call sequences."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

from vectorpro.host import HOST_TYPES, MemoryHostContext
from vectorpro.learning.plan import LearningPlan, OutputWidth
from vectorpro.learning.registry import Capability, Registry, program_provenance
from vectorpro.machine import Instr, assemble


@dataclass
class StateExample:
    inputs: tuple[str | int, ...]
    before: dict[str, bytes]
    after: dict[str, bytes]
    output: int | None = None

    def signature(self):
        return (self.inputs, tuple(sorted(self.before.items())))

@dataclass
class StateLesson:
    input_types: tuple[str, ...]
    training: list[StateExample]
    validation: list[StateExample]
    width: int = 16
    max_steps: int = 3
    candidate_budget: int = 5000

    def validate(self):
        if not self.input_types or any(t not in ("path", "value") for t in self.input_types):
            raise ValueError("state inputs must declare path/value types")
        if self.width < 1 or self.max_steps < 1 or self.candidate_budget < 1:
            raise ValueError("state widths and search budgets must be positive")
        if not self.training or not self.validation:
            raise ValueError("state learning requires training and validation examples")
        for case in self.training + self.validation:
            if len(case.inputs) != len(self.input_types):
                raise ValueError("state input arity mismatch")
            for value, kind in zip(case.inputs, self.input_types):
                if kind == "path":
                    if not isinstance(value, str):
                        raise ValueError("path inputs must be strings")
                    MemoryHostContext.normalize(value)
                elif type(value) is not int or not 0 <= value < 1 << self.width:
                    raise ValueError("numeric state inputs must fit register width")
            for files in (case.before, case.after):
                for name, data in files.items():
                    if MemoryHostContext.normalize(name) != name or not isinstance(data, bytes):
                        raise ValueError("state files require normalized paths and bytes")
            if case.output is not None and (type(case.output) is not int or not 0 <= case.output < 1 << self.width):
                raise ValueError("state output must fit register width")
        train = [c.signature() for c in self.training]
        validation = [c.signature() for c in self.validation]
        if len(set(train)) != len(train) or len(set(validation)) != len(validation) or set(train) & set(validation):
            raise ValueError("state training/validation initial conditions must be distinct")

    @classmethod
    def from_dict(cls, data):
        def cases(name):
            return [StateExample(tuple(c["inputs"]),
                                 {p: bytes.fromhex(v) for p, v in c["before"].items()},
                                 {p: bytes.fromhex(v) for p, v in c["after"].items()}, c.get("output"))
                    for c in data[name]]
        return cls(tuple(data["input_types"]), cases("training"), cases("validation"),
                   data.get("width", 16), data.get("max_steps", 3), data.get("candidate_budget", 5000))


@dataclass
class StateOutcome:
    capability: Capability | None
    history: list[dict] = field(default_factory=list)


def evaluate(executable, case: StateExample, lesson: StateLesson) -> bool:
    context = MemoryHostContext(case.before)
    inputs = tuple(context.put(value.encode("utf-8")) if kind == "path" else value
                   for value, kind in zip(case.inputs, lesson.input_types))
    try:
        with context.activate():
            output = executable([inputs], lesson.width)[0]
        return context.files == case.after and (case.output is None or output == case.output)
    except (OSError, ValueError, KeyError, IndexError, RuntimeError):
        return False


def learn_stateful(registry: Registry, name: str, lesson: StateLesson) -> StateOutcome:
    lesson.validate()
    if name in registry:
        raise ValueError("cannot replace an existing state capability")
    # Only generic host primitives actually provided by this machine are available.
    operations = [(cap.name, HOST_TYPES[cap.provenance["operation"]]) for cap in registry
                  if cap.provenance.get("kind") == "host"]
    plan = LearningPlan(name, "learned state transformation", len(lesson.input_types), OutputWidth.SAME)
    initial = {f"x{i}": "zero" for i in range(plan.arity)} | {"zero": "zero", "one": "one"}
    types = {f"x{i}": t for i, t in enumerate(lesson.input_types)} | {"zero": "value", "one": "value"}
    frontier = [((), types)]
    tried = 0
    for depth in range(1, lesson.max_steps + 1):
        next_frontier = []
        for prefix, available in frontier:
            for op, (arguments, result_type) in operations:
                choices = [[r for r, t in available.items() if t == kind] for kind in arguments]
                for args in product(*choices):
                    if tried >= lesson.candidate_budget:
                        return StateOutcome(None, [{"candidates": tried, "accepted": False, "reason": "candidate budget"}])
                    tried += 1
                    destination = f"t{depth}"
                    instructions = prefix + (Instr(op, tuple(args), destination),)
                    registers = initial | {f"t{i}": "zero" for i in range(1, depth + 1)}
                    program = assemble(instructions, [f"x{i}" for i in range(plan.arity)],
                                       registers, destination, registry.key_of)
                    provenance = program_provenance(program, "state example search",
                                                    input_types=list(lesson.input_types),
                                                    teaching_width=lesson.width)
                    executable = registry.build(plan, provenance)
                    if all(evaluate(executable, c, lesson) for c in lesson.training):
                        if all(evaluate(executable, c, lesson) for c in lesson.validation):
                            history = [{"strategy": "state search", "steps": depth, "candidates": tried,
                                        "training_cases": len(lesson.training), "validation_cases": len(lesson.validation),
                                        "accepted": True}]
                            capability = Capability(plan, executable, provenance, history)
                            registry.add(capability)
                            return StateOutcome(capability, history)
                    next_frontier.append((instructions, available | {destination: result_type}))
        frontier = next_frontier
    return StateOutcome(None, [{"candidates": tried, "accepted": False, "reason": "step budget"}])
