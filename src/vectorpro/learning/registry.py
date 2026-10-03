"""The capability registry: learned vector programs, how they came to be, and what they can do.

Every capability is stored as data only: a learned unit as its schema spec
plus binary table, a composition as an expression over other capabilities.
The registry can be searched by text or by behaviour, explained in plain
language, and asked to run a capability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import torch

from vectorpro.cells import CellSignature, TableCell
from vectorpro.cells.base import binary_domain
from vectorpro.execution import BitExecutable
from vectorpro.expr import expr_from_data, operators, render
from vectorpro.learning.examples import ExampleSet, Operands
from vectorpro.learning.plan import LearningPlan, OutputWidth
from vectorpro.programs import ExpressionProgram, Window
from vectorpro.schemas import schema_from_spec
from vectorpro.units import FunctionUnit

COMPOSABLE_OUTPUTS = (OutputWidth.SAME, OutputWidth.PLUS_ONE, OutputWidth.DOUBLE)


@dataclass
class Capability:
    plan: LearningPlan
    executable: BitExecutable
    provenance: dict
    history: list[dict] = field(default_factory=list)
    audit: dict | None = None

    @property
    def name(self) -> str:
        return self.plan.name

    def run(self, operand_tuples: Sequence[Operands], width: int) -> list[int]:
        return self.executable(operand_tuples, width)

    def describe(self) -> str:
        plan = self.plan
        lines = [
            f"{plan.name}: {plan.description}",
            f"  shape: {plan.arity} operand(s) of W bits -> {plan.output.value} bits",
        ]
        prov = self.provenance
        if prov["kind"] == "composition":
            lines.append(f"  built by composing: {render(expr_from_data(prov['expression']))}")
            lines.append(f"  uses: {', '.join(prov['uses'])}")
        else:
            lines.append(f"  learned unit: {describe_schema(prov['schema'])}")
            lines.append("  local rule (inputs | state -> outputs | next state):")
            lines += [f"    {row}" for row in describe_table(prov)]
        if self.history:
            last = self.history[-1]
            lines.append(
                f"  learned in {len(self.history)} round(s) from {last['train_examples']} examples; "
                f"validation {last['validation_accuracy']:.0%} at {plan.validation_width} bits"
            )
        return "\n".join(lines)


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


def composition_provenance(expr_data: dict) -> dict:
    expr = expr_from_data(expr_data)
    return {"kind": "composition", "expression": expr_data, "uses": sorted(operators(expr))}


class Registry:
    def __init__(self) -> None:
        self._caps: dict[str, Capability] = {}

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
        self._caps[capability.name] = capability

    # -- building executables from data -------------------------------------------

    def build(self, plan: LearningPlan, provenance: dict) -> BitExecutable:
        if provenance["kind"] == "unit":
            sig = CellSignature(*provenance["signature"])
            table = TableCell(sig, torch.tensor(provenance["table"]))
            return FunctionUnit(plan.name, schema_from_spec(provenance["schema"]), table)
        expr = expr_from_data(provenance["expression"])
        return ExpressionProgram(expr, {n: self.operator(n) for n in operators(expr)}, plan.arity)

    def operator(self, name: str) -> BitExecutable:
        """``name`` as a ``W -> W`` binary operator (low word of its result)."""
        cap = self.get(name)
        if cap.plan.arity != 2 or cap.plan.output not in COMPOSABLE_OUTPUTS:
            raise ValueError(f"{name} is not usable as a W-bit binary operator")
        return Window(cap.executable)

    def operators(self) -> dict[str, BitExecutable]:
        return {
            c.name: self.operator(c.name)
            for c in self
            if c.plan.arity == 2 and c.plan.output in COMPOSABLE_OUTPUTS
        }

    # -- finding and using capabilities -------------------------------------------

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
        return [c for c in self if c.plan.arity == arity and examples.accuracy(c.run) == 1.0]

    def run(self, name: str, operand_tuples: Sequence[Operands], width: int) -> list[int]:
        return self.get(name).run(operand_tuples, width)

    def describe(self) -> str:
        return "\n\n".join(c.describe() for c in self)

    # -- persistence -----------------------------------------------------------------

    def save(self, path: Path) -> None:
        records = [
            {"plan": c.plan.to_dict(), "provenance": c.provenance, "history": c.history, "audit": c.audit}
            for c in self
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"capabilities": records}, indent=1))

    @classmethod
    def load(cls, path: Path) -> Registry:
        registry = cls()
        for record in json.loads(path.read_text())["capabilities"]:
            plan = LearningPlan.from_dict(record["plan"])
            executable = registry.build(plan, record["provenance"])
            registry.add(Capability(plan, executable, record["provenance"], record["history"], record["audit"]))
        return registry
