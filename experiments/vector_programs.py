"""M2b: all program logic in vectors, run by one task-agnostic kernel.

1. Learn units from examples only (and, or, xor, add, sub, lt, shl, shr).
2. Register hand-authored loop/branch programs (mul, div, mod) as vector
   programs that call the *learned* units by address vector.
3. Learn compositions (square, mul_add, sum_mod, diff_div) from examples
   only. The learner finds them by reuse, stored as vector programs that call
   the loop programs: functions calling functions, all as tensors.
4. Controls:
   * every capability is evaluated on 8/16/32-bit inputs;
   * swapping two units' address vectors must break programs that call them;
   * every single re-routing of the mul program's tensors (a read, a write or
     a jump) is applied in turn, and must change its behaviour;
   * the kernel and program-format sources contain no capability names;
   * the saved registry reloads and gives identical results.

Usage: python experiments/vector_programs.py [--seeds ...]
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

import torch

import vectorpro.machine.kernel as kernel_module
import vectorpro.machine.program as program_module
from authored import divmod_program, mul
from targets import TARGETS
from vectorpro.data import random_tuples
from vectorpro.learning import Capability, ExampleSet, Learner, LearningPlan, Registry
from vectorpro.learning.registry import program_provenance
from vectorpro.machine import BudgetExceeded, ProgramExecutable, VectorProgram

HERE = Path(__file__).parent
EVAL_WIDTHS, EVAL_SAMPLES = (8, 16, 32), 200
AUTHORED = {
    "mul": mul,
    "div": lambda key_of: divmod_program(key_of, "q"),
    "mod": lambda key_of: divmod_program(key_of, "r"),
}


def eval_sets(name: str, arity: int, seed: int) -> list[ExampleSet]:
    rng = random.Random(f"eval-{name}-{seed}")
    sets = []
    for width in EVAL_WIDTHS:
        ops = random_tuples(arity, width, EVAL_SAMPLES, rng)
        # Small operands too, so zero divisors and short loops occur.
        ops += random_tuples(arity, 3, 40, rng)
        sets.append(ExampleSet(width, ops, [TARGETS[name](o, width) for o in ops]))
    return sets


def accuracy(run, sets: list[ExampleSet]) -> dict:
    out = {}
    for s in sets:
        try:
            out[f"{s.width}bit"] = s.accuracy(run)
        except BudgetExceeded:
            out[f"{s.width}bit"] = 0.0
    return out


def reroutings(program: VectorProgram, n_args: list[int]):
    """Every program differing in exactly one routing choice: a read, a write or a jump."""
    n_regs, n_steps = program.n_registers, program.n_steps

    def moved(tensor: torch.Tensor, index: tuple, size: int):
        current = int(tensor[index].argmax())
        for alt in range(size):
            if alt != current:
                t = tensor.clone()
                t[index] = 0
                t[index + (alt,)] = 1
                yield alt, t

    for s in range(n_steps):
        for k in range(n_args[s]):
            for alt, t in moved(program.reads, (s, k), n_regs):
                yield f"@{s} arg{k} reads r{alt}", replace(program, reads=t)
        if n_args[s]:
            for alt, t in moved(program.writes, (s,), n_regs):
                yield f"@{s} writes r{alt}", replace(program, writes=t)
        for field in ("next_true", "next_false"):
            if field == "next_false" and int(program.cond[s].argmax()) == n_regs:
                continue  # unconditional step: next_false is never taken
            for alt, t in moved(getattr(program, field), (s,), n_steps + 1):
                yield f"@{s} {field} -> {alt}", replace(program, **{field: t})


def rerouting_control(registry: Registry, name: str, seed: int) -> dict:
    program = registry.get(name).executable.program
    n_args = [0 if program.writes[s, -1] > 0.5 else registry.resolve(program.keys[s]).arity
              for s in range(program.n_steps)]
    sets = eval_sets(name, 2, seed)[:1]  # 8-bit
    changed, total, unchanged = 0, 0, []
    for label, variant in reroutings(program, n_args):
        total += 1
        executable = ProgramExecutable(variant, registry, budget=2000)
        if accuracy(executable, sets)["8bit"] < 1.0:
            changed += 1
        else:
            unchanged.append(label)
    return {"program": name, "reroutings": total, "behaviour_changed": changed, "unchanged": unchanged}


def key_swap_control(registry: Registry, seed: int) -> dict:
    add, sub = registry.get("add"), registry.get("sub")
    before = accuracy(registry.get("mul").executable, eval_sets("mul", 2, seed)[:1])["8bit"]
    add.key, sub.key = sub.key, add.key
    after = accuracy(registry.get("mul").executable, eval_sets("mul", 2, seed)[:1])["8bit"]
    add.key, sub.key = sub.key, add.key
    return {"mul_8bit_before": before, "mul_8bit_with_add_sub_keys_swapped": after}


def kernel_is_task_agnostic(names: list[str]) -> dict:
    sources = {m.__name__: Path(m.__file__).read_text() for m in (kernel_module, program_module)}
    hits = {mod: [n for n in names if f'"{n}"' in src or f"'{n}'" in src] for mod, src in sources.items()}
    return {"capability_names_in_kernel_sources": hits, "clean": not any(hits.values())}


def run_seed(curriculum: dict, seed: int, out_dir: Path) -> dict:
    torch.manual_seed(seed)
    registry = Registry(seed=seed)
    learner = Learner(registry)
    record: dict = {"seed": seed, "capabilities": {}}

    def learn(spec: dict, stage: str) -> None:
        plan = LearningPlan.from_dict(spec)
        outcome = learner.learn(plan, TARGETS[plan.name], random.Random(f"{plan.name}-{seed}"))
        last = outcome.history[-1]
        entry = {"stage": stage, "learned": outcome.learned, "strategy": last["strategy"],
                 "structure": last["structure"], "examples": last["train_examples"]}
        if outcome.learned:
            entry["accuracy"] = accuracy(outcome.capability.executable, eval_sets(plan.name, plan.arity, seed))
        record["capabilities"][plan.name] = entry
        print(f"  {plan.name:<9} {stage:<11} {'ok' if outcome.learned else 'FAILED':<6} "
              f"{last['strategy']:<5} {last['train_examples']:>2} ex  {entry.get('accuracy')}", flush=True)

    for spec in curriculum["units"]:
        learn(spec, "unit")
    for spec in curriculum["authored"]:
        plan = LearningPlan.from_dict(spec)
        provenance = program_provenance(AUTHORED[plan.name](registry.key_of), "authored")
        registry.add(Capability(plan, registry.build(plan, provenance), provenance))
        acc = accuracy(registry.get(plan.name).executable, eval_sets(plan.name, plan.arity, seed))
        record["capabilities"][plan.name] = {"stage": "authored", "learned": None, "accuracy": acc}
        print(f"  {plan.name:<9} authored    -      -            {acc}", flush=True)
    for spec in curriculum["compositions"]:
        learn(spec, "composition")

    record["controls"] = {
        "key_swap": key_swap_control(registry, seed),
        "rerouting": rerouting_control(registry, "mul", seed),
        "kernel": kernel_is_task_agnostic([c.name for c in registry]),
    }
    path = out_dir / f"vector_registry_{seed}.json"
    registry.save(path)
    reloaded = Registry.load(path)
    probe = random_tuples(3, 16, 50, random.Random(seed))
    record["controls"]["reload_identical"] = all(
        reloaded.run(c.name, [t[: c.plan.arity] for t in probe], 16)
        == registry.run(c.name, [t[: c.plan.arity] for t in probe], 16)
        for c in registry
    )
    record["explain"] = {n: registry.explain(n) for n in ("mul", "mul_add") if n in registry}
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--curriculum", type=Path, default=HERE / "curricula" / "machine.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261002, 20261003, 20261004])
    parser.add_argument("--out", type=Path, default=Path("results/vector_programs.json"))
    args = parser.parse_args()

    curriculum = json.loads(args.curriculum.read_text())
    runs = []
    for seed in args.seeds:
        print(f"seed {seed}", flush=True)
        runs.append(run_seed(curriculum, seed, args.out.parent))
        c = runs[-1]["controls"]
        print(f"  controls: key swap {c['key_swap']}, rerouting "
              f"{c['rerouting']['behaviour_changed']}/{c['rerouting']['reroutings']} changed behaviour, "
              f"kernel clean {c['kernel']['clean']}, reload identical {c['reload_identical']}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"experiment": "vector_programs", "runs": runs}, indent=1))
    print("\n" + runs[0]["explain"].get("mul_add", ""))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
