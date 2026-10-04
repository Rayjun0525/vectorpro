"""The capability registry: vector programs, how they came to be, and what they can do.

Every capability has an **address vector**; programs call capabilities by
address, never by name. A capability is stored as data only:

* ``unit``:    a learned schema spec plus its binary table;
* ``program``: a vector program (tensors) whose steps address other capabilities.

The registry resolves address vectors (it is the kernel's resolver), can be
searched by text or by behaviour, explains capabilities in plain language
from their tensors, runs them, and saves and loads them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import torch
import torch.nn.functional as F

from vectorpro.cells import CellSignature, TableCell
from vectorpro.cells.base import binary_domain
from vectorpro.execution import BitExecutable, Fitted
from vectorpro.learning.examples import ExampleSet, Operands
from vectorpro.learning.plan import LearningPlan
from vectorpro.learning.search import Operator, detect_commutative
from vectorpro.machine import ProgramExecutable, VectorProgram, disassemble
from vectorpro.schemas import schema_from_spec
from vectorpro.units import FunctionUnit

@dataclass
class Capability:
    plan: LearningPlan
    executable: BitExecutable
    provenance: dict
    history: list[dict] = field(default_factory=list)
    audit: dict | None = None
    key: torch.Tensor | None = None  # address vector, assigned on registration

    @property
    def name(self) -> str:
        return self.plan.name

    def run(self, operand_tuples: Sequence[Operands], width: int) -> list[int]:
        return self.executable(operand_tuples, width)


def describe_schema(spec: dict) -> str:
    if spec["kind"] == "map":
        return f"same rule applied to every bit position independently ({spec['arity']} inputs per bit)"
    order = "lowest to highest bit" if spec["direction"] == "lsb_first" else "highest to lowest bit"
    tail = ", then emits its final state" if spec["emit_final_state"] else ""
    return (f"walks bits {order} carrying {len(spec['initial_state'])} state bit(s) "
            f"(starting at {spec['initial_state']}){tail}")


def describe_table(prov: dict) -> list[str]:
    sig = CellSignature(*prov["signature"])
    rows = []
    for bits, out in zip(binary_domain(sig).long().tolist(), prov["table"]):
        ins, state = bits[: sig.n_inputs], bits[sig.n_inputs :]
        y, nxt = out[: sig.n_outputs], out[sig.n_outputs :]
        rows.append(f"{''.join(map(str, ins))} | {''.join(map(str, state)) or '-'} -> "
                    f"{''.join(map(str, y)) or '-'} | {''.join(map(str, nxt)) or '-'}")
    return rows


def unit_provenance(unit: FunctionUnit) -> dict:
    table = TableCell.from_cell(unit.cell)
    sig = table.signature
    return {
        "kind": "unit",
        "schema": unit.schema.to_spec(),
        "signature": [sig.n_inputs, sig.n_state, sig.n_outputs],
        "table": table.table.long().tolist(),
    }


def program_provenance(program: VectorProgram, origin: str, **notes) -> dict:
    """``notes`` record how the program was found (e.g. the fold it came from).

    They are descriptive and reusable as search material; execution reads only ``program``.
    """
    return {"kind": "program", "origin": origin, "program": program.to_data(), **notes}


class Registry:
    def __init__(self, key_dim: int = 32, seed: int = 0) -> None:
        self._caps: dict[str, Capability] = {}
        self._operator_cache: dict[str, Operator] = {}
        self.has_effects = False
        self.key_dim = key_dim
        self._rng = torch.Generator().manual_seed(seed)

    def __contains__(self, name: str) -> bool:
        return name in self._caps

    def __iter__(self) -> Iterator[Capability]:
        return iter(self._caps.values())

    def __len__(self) -> int:
        return len(self._caps)

    def get(self, name: str) -> Capability:
        return self._caps[name]

    def add(self, capability: Capability) -> None:
        if capability.name in self._caps:
            raise ValueError(f"capability {capability.name!r} already registered")
        if capability.key is None:
            capability.key = F.normalize(torch.randn(self.key_dim, generator=self._rng), dim=0)
        self._caps[capability.name] = capability
        # Programs built by this registry can only call capabilities held here.
        # Avoid walking every pure program tree during large arithmetic searches.
        if not isinstance(capability.executable, ProgramExecutable):
            self.has_effects = self.has_effects or capability.executable.effects
        elif capability.executable.resolver is not self:
            self.has_effects = self.has_effects or capability.executable.effects

    # -- addressing (the kernel's resolver) ------------------------------------------

    def key_of(self, name: str) -> torch.Tensor:
        return self.get(name).key

    def _nearest(self, key: torch.Tensor) -> Capability:
        caps = list(self)
        keys = torch.stack([c.key for c in caps])
        return caps[int((keys @ F.normalize(key, dim=0)).argmax())]

    def resolve(self, key: torch.Tensor) -> BitExecutable:
        return self._nearest(key).executable

    def name_of(self, key: torch.Tensor) -> str:
        return self._nearest(key).name

    # -- building executables from data -----------------------------------------------

    def build(self, plan: LearningPlan, provenance: dict) -> BitExecutable:
        if provenance["kind"] == "unit":
            sig = CellSignature(*provenance["signature"])
            table = TableCell(sig, torch.tensor(provenance["table"]))
            return FunctionUnit(plan.name, schema_from_spec(provenance["schema"]), table)
        if provenance["kind"] == "program":
            return ProgramExecutable(VectorProgram.from_data(provenance["program"]), self)
        if provenance["kind"] == "host":
            from vectorpro.host import HostExecutable
            return HostExecutable(provenance["operation"])
        raise ValueError(f"unknown provenance kind {provenance['kind']!r}")

    def loops(self, name: str) -> bool:
        """Whether running ``name`` can loop: its own control tensors, or any capability it calls."""
        executable = self.get(name).executable
        if not isinstance(executable, ProgramExecutable):
            return False
        program = executable.program
        if program.has_loop:
            return True
        called = {self.name_of(program.keys[s]) for s in range(program.n_steps)
                  if program.calls[s] > 0.5}
        return any(self.loops(n) for n in called if n != name)

    def operators(self) -> list[Operator]:
        """Every capability as a ``W -> W`` search operator (results fitted to the operand width)."""
        ops = []
        for cap in self:
            if self.has_effects and cap.executable.effects:
                continue  # never execute effects while enumerating learning candidates
            if cap.name not in self._operator_cache:
                executable = Fitted(cap.executable)
                commutative = cap.plan.arity == 2 and detect_commutative(executable)
                program = isinstance(cap.executable, ProgramExecutable)
                loops = self.loops(cap.name)
                cost = 100 if loops else 3 if program else 1  # loop programs are costly to run in bulk
                self._operator_cache[cap.name] = Operator(
                    cap.name, executable, cap.plan.arity, cost, commutative, loops)
            ops.append(self._operator_cache[cap.name])
        return ops

    # -- finding, explaining and using capabilities ---------------------------------

    def search(self, query: str) -> list[Capability]:
        """Capabilities whose name, description or tags contain every query word."""
        words = query.lower().split()
        found = []
        for cap in self:
            text = " ".join([cap.name, cap.plan.description, *cap.plan.tags]).lower()
            if all(w in text for w in words):
                found.append(cap)
        return found

    def find_by_examples(self, arity: int, examples: ExampleSet) -> list[Capability]:
        """Capabilities that reproduce every given example."""
        return [c for c in self if (not self.has_effects or not c.executable.effects) and c.plan.arity == arity
                and examples.accuracy(c.run) == 1.0]

    def run(self, name: str, operand_tuples: Sequence[Operands], width: int) -> list[int]:
        return self.get(name).run(operand_tuples, width)

    def explain(self, name: str) -> str:
        """Plain-language description, reconstructed from the stored tensors."""
        cap = self.get(name)
        plan, prov = cap.plan, cap.provenance
        lines = [
            f"{plan.name}: {plan.description}",
            f"  shape: {plan.arity} operand(s) of W bits -> {plan.output.value} bits",
        ]
        if prov["kind"] == "program":
            lines.append(f"  vector program ({prov['origin']}), disassembled from its tensors:")
            program = VectorProgram.from_data(prov["program"])
            lines += [f"    {line}" for line in disassemble(program, self)]
        elif prov["kind"] == "host":
            lines.append(f"  host execution primitive: {prov['operation']} (provided, not learned)")
        else:
            lines.append(f"  learned unit: {describe_schema(prov['schema'])}")
            lines.append("  local rule (inputs | state -> outputs | next state):")
            lines += [f"    {row}" for row in describe_table(prov)]
        if cap.history:
            last = cap.history[-1]
            if last.get("strategy") == "state search":
                lines.append(f"  learned from {last['training_cases']} training cases and "
                             f"{last['validation_cases']} validation cases; "
                             f"{last['steps']} steps, {last['candidates']} candidates")
            else:
                lines.append(
                    f"  learned in {len(cap.history)} round(s) from {last['train_examples']} examples; "
                    f"validation {last['validation_accuracy']:.0%} at {plan.validation_width} bits"
                )
        return "\n".join(lines)

    def describe(self) -> str:
        return "\n\n".join(self.explain(c.name) for c in self)

    # -- persistence -------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_data(), indent=1), encoding="utf-8")

    def to_data(self) -> dict:
        records = [
            {"plan": c.plan.to_dict(), "key": c.key.tolist(), "provenance": c.provenance,
             "history": c.history, "audit": c.audit}
            for c in self
        ]
        return {"key_dim": self.key_dim, "capabilities": records,
                "rng_state": self._rng.get_state().tolist()}

    @classmethod
    def load(cls, path: Path) -> Registry:
        return cls.from_data(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def from_data(cls, data: dict) -> Registry:
        registry = cls(key_dim=data["key_dim"])
        for record in data["capabilities"]:
            plan = LearningPlan.from_dict(record["plan"])
            executable = registry.build(plan, record["provenance"])
            registry.add(Capability(plan, executable, record["provenance"], record["history"],
                                    record["audit"], torch.tensor(record["key"])))
        if "rng_state" in data:
            registry._rng.set_state(torch.tensor(data["rng_state"], dtype=torch.uint8))
        else:
            for _ in registry:
                torch.randn(registry.key_dim, generator=registry._rng)
        return registry
