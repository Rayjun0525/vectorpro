"""Controlled side experiment. Read PROTOCOL.md before interpreting results."""
from __future__ import annotations

import ctypes
import hashlib
import itertools
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

SEED = 20261004
OPS = ("add", "sub", "mul", "div")
ALPHA = 0.01
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "binary_embedding"
BUILD = Path("/tmp/vectorpro-binary-embedding")
PIPELINES = (("xor",), ("add",), ("rot",), ("mul",),
             ("xor", "add"), ("rot", "xor"), ("add", "rot"), ("mul", "xor"))


def command(*args: str) -> str:
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def wrap(variable: str, pipeline: tuple[str, ...], constants: list[int]) -> str:
    lines = []
    for kind, constant in zip(pipeline, constants):
        if kind == "xor":
            expression = f"{variable} ^ {constant}u8"
        elif kind == "add":
            expression = f"{variable}.wrapping_add({constant}u8)"
        elif kind == "rot":
            expression = f"{variable}.rotate_left({constant % 7 + 1})"
        else:
            expression = f"{variable}.wrapping_mul({constant | 1}u8)"
        lines.append(f"let {variable} = black_box({expression});")
    for kind, constant in reversed(list(zip(pipeline, constants))):
        if kind == "xor":
            expression = f"{variable} ^ {constant}u8"
        elif kind == "add":
            expression = f"{variable}.wrapping_sub({constant}u8)"
        elif kind == "rot":
            expression = f"{variable}.rotate_right({constant % 7 + 1})"
        else:
            expression = f"{variable}.wrapping_mul({pow(constant | 1, -1, 256)}u8)"
        lines.append(f"let {variable} = {expression};")
    return "\n".join(lines)


def source(template: int, constants: list[list[int]], operation: str) -> str:
    expression = {"add": "a.wrapping_add(b)", "sub": "a.wrapping_sub(b)",
                  "mul": "a.wrapping_mul(b)",
                  "div": "if b == 0 { 255u8 } else { a / b }"}[operation]
    pipeline = PIPELINES[template]
    return ("#![no_std]\nuse core::hint::black_box;\n#[no_mangle]\n"
            "pub extern \"C\" fn f(a: u8, b: u8) -> u8 {\n"
            + wrap("a", pipeline, constants[0]) + "\n"
            + wrap("b", pipeline, constants[1]) + "\n"
            + f"let z = {expression};\n"
            + wrap("z", pipeline, constants[2]) + "\nz\n}\n")


def reference(op: str, a: int, b: int) -> int:
    if op == "add":
        return (a + b) & 255
    if op == "sub":
        return (a - b) & 255
    if op == "mul":
        return (a * b) & 255
    return a // b if b else 255


def embedding(code: bytes) -> np.ndarray:
    values = np.frombuffer(code, dtype=np.uint8).astype(np.int64)
    histogram = np.bincount(values, minlength=256) / len(values)
    pairs = (values[:-1] * 257 + values[1:]) % 1024
    bigrams = np.bincount(pairs, minlength=1024) / max(1, len(pairs))
    result = np.concatenate([histogram, bigrams])
    return result / np.linalg.norm(result)


def collect() -> tuple[list[dict], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    records, vectors, tables = [], [], []
    for template in range(8):
        for variant in range(4):
            constants = rng.integers(1, 255, size=(3, len(PIPELINES[template]))).tolist()
            for label, op in enumerate(OPS):
                index = len(records)
                stem = BUILD / f"sample_{index:03d}"
                src, obj, lib, raw = [stem.with_suffix(s) for s in (".rs", ".o", ".so", ".bin")]
                code_source = source(template, constants, op)
                src.write_text(code_source)
                opt = 1 if template < 4 else 3
                command("rustc", str(src), "--crate-name", "sample", "--crate-type", "lib",
                        "--edition", "2021", "--emit=obj", "-C", f"opt-level={opt}",
                        "-C", "panic=abort", "-C", "overflow-checks=off",
                        "-C", "relocation-model=pic", "-C", "target-cpu=x86-64",
                        "-o", str(obj))
                # No unresolved helper calls: all extracted code must be self-contained.
                undefined = command("nm", "-u", str(obj))
                if undefined:
                    raise RuntimeError(f"unresolved symbols in {index}: {undefined}")
                command("objcopy", "--dump-section", f".text.f={raw}", str(obj))
                code = raw.read_bytes()
                if not code:
                    raise RuntimeError("empty function code")
                command("gcc", "-shared", "-Wl,-z,defs", str(obj), "-o", str(lib))
                library = ctypes.CDLL(str(lib))
                function = library.f
                function.argtypes = [ctypes.c_uint8, ctypes.c_uint8]
                function.restype = ctypes.c_uint8
                table = []
                for a in range(256):
                    for b in range(256):
                        actual = function(a, b)
                        expected = reference(op, a, b)
                        if actual != expected:
                            raise AssertionError((index, a, b, actual, expected))
                        if a < 16 and b < 16:
                            table.append(actual)
                records.append(dict(index=index, template=template, variant=variant,
                                    family=template * 4 + variant, label=label, operation=op,
                                    split="train" if template < 4 else "test", opt_level=opt,
                                    byte_length=len(code), sha256=hashlib.sha256(code).hexdigest(),
                                    code_hex=code.hex(), constants=constants, source=code_source))
                vectors.append(embedding(code))
                tables.append(table)
            print(f"compiled and exhaustively checked family {template * 4 + variant + 1}/32", flush=True)
    train_hashes = {r["sha256"] for r in records if r["split"] == "train"}
    test_hashes = {r["sha256"] for r in records if r["split"] == "test"}
    if train_hashes & test_hashes:
        raise AssertionError("identical code leaked across train/test")
    return records, np.array(vectors), np.array(tables, dtype=np.uint8)


def ridge_operator(train: np.ndarray, test: np.ndarray):
    mean = train.mean(axis=0)
    centered = train - mean
    inverse = np.linalg.solve(centered @ centered.T + ALPHA * np.eye(len(train)), np.eye(len(train)))
    # A test-to-training linear operator, including the intercept.
    transfer = (test - mean) @ centered.T @ inverse
    transfer += (1 - transfer.sum(axis=1))[:, None] / len(train)
    return transfer, mean, centered, inverse


def exact_block_permutation(predictions: np.ndarray, labels: np.ndarray, templates: np.ndarray) -> dict:
    distribution = np.array([1], dtype=np.int64)
    for template in sorted(set(templates.tolist())):
        mask = templates == template
        scores = [int(np.sum(predictions[mask] == np.array(p)[labels[mask]]))
                  for p in itertools.permutations(range(4))]
        distribution = np.convolve(distribution, np.bincount(scores, minlength=int(mask.sum()) + 1))
    observed = int(np.sum(predictions == labels))
    total = int(distribution.sum())
    p = float(distribution[observed:].sum() / total)
    return dict(observed_correct=observed, permutations=total, one_sided_p=p,
                null="operator-label permutations shared within each wrapper-template block")


def classifier_metrics(predictions: np.ndarray, labels: np.ndarray, templates: np.ndarray) -> dict:
    confusion = np.zeros((4, 4), dtype=int)
    np.add.at(confusion, (labels, predictions), 1)
    per_template = {str(t): float(np.mean(predictions[templates == t] == labels[templates == t]))
                    for t in sorted(set(templates.tolist()))}
    rng = np.random.default_rng(SEED + 1)
    block_accuracies = np.array(list(per_template.values()))
    boot = rng.choice(block_accuracies, size=(20000, len(block_accuracies))).mean(axis=1)
    return dict(accuracy=float(np.mean(predictions == labels)),
                correct=int(np.sum(predictions == labels)), total=len(labels),
                confusion_matrix_true_rows_predicted_columns=confusion.tolist(),
                operation_order=list(OPS), per_template_accuracy=per_template,
                template_bootstrap_95_percent_interval=np.quantile(boot, [0.025, 0.975]).tolist(),
                permutation_test=exact_block_permutation(predictions, labels, templates))


def table_metrics(predicted_bits: np.ndarray, target_tables: np.ndarray) -> dict:
    values = (predicted_bits.astype(np.uint16) * (1 << np.arange(8))).sum(axis=2)
    exact = values == target_tables
    true_bits = ((target_tables[:, :, None] >> np.arange(8)) & 1).astype(bool)
    return dict(exact_output_accuracy=float(exact.mean()), correct_outputs=int(exact.sum()),
                total_outputs=int(exact.size), bit_accuracy=float(np.mean(predicted_bits == true_bits)),
                fully_correct_tables=int(np.sum(exact.all(axis=1))), total_tables=len(exact),
                scope="inputs 0..15; all these numeric inputs appear in training tables")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)
    records, vectors, tables = collect()
    (OUT / "corpus.json").write_text(json.dumps(records, indent=2) + "\n")
    train = np.array([r["split"] == "train" for r in records])
    labels = np.array([r["label"] for r in records])
    test_templates = np.array([r["template"] for r in records])[~train]
    targets = np.eye(4)[labels[train]]
    transfer, mean, centered, inverse = ridge_operator(vectors[train], vectors[~train])
    scores = transfer @ targets
    predictions = scores.argmax(axis=1)

    lengths = np.log(np.array([r["byte_length"] for r in records]))[:, None]
    lengths = (lengths - lengths[train].mean()) / lengths[train].std()
    length_transfer, _, _, _ = ridge_operator(lengths[train], lengths[~train])
    length_predictions = (length_transfer @ targets).argmax(axis=1)

    bits = ((tables[:, :, None] >> np.arange(8)) & 1).astype(float)
    flat_targets = bits[train].reshape(int(train.sum()), -1) * 2 - 1
    decoded = (transfer @ flat_targets).reshape(-1, 256, 8) > 0
    length_decoded = (length_transfer @ flat_targets).reshape(-1, 256, 8) > 0

    rng = np.random.default_rng(SEED + 2)
    negative = []
    for _ in range(1000):
        random_labels = np.concatenate([rng.permutation(4) for _ in range(16)])
        random_predictions = (transfer @ np.eye(4)[random_labels]).argmax(axis=1)
        negative.append(float(np.mean(random_predictions == labels[~train])))

    # Save an executable finite-table predictor: embedding -> learned bits -> table lookup.
    # It never dispatches to handwritten add/sub/mul/div functions.
    bit_mean = flat_targets.mean(axis=0)
    bit_dual = inverse @ (flat_targets - bit_mean)
    class_mean = targets.mean(axis=0)
    class_dual = inverse @ (targets - class_mean)
    np.savez_compressed(OUT / "model_and_predictions.npz", embedding_mean=mean,
                        training_embedding_basis=centered, output_bit_mean=bit_mean,
                        output_bit_dual_weights=bit_dual, class_mean=class_mean,
                        class_dual_weights=class_dual, all_embeddings=vectors,
                        heldout_predictions=predictions, heldout_predicted_bits=decoded,
                        heldout_target_tables=tables[~train], heldout_embeddings=vectors[~train])
    with np.load(OUT / "model_and_predictions.npz") as saved:
        restored = (((saved["heldout_embeddings"] - saved["embedding_mean"])
                     @ saved["training_embedding_basis"].T @ saved["output_bit_dual_weights"]
                     + saved["output_bit_mean"]).reshape(-1, 256, 8) > 0)
        if not np.array_equal(restored, decoded):
            raise AssertionError("saved tensor predictor does not reproduce results")

    result = dict(
        experiment="binary_embedding_side_experiment", seed=SEED,
        environment=dict(rustc=command("rustc", "--version"), gcc=command("gcc", "-dumpfullversion"),
                         python=platform.python_version(), numpy=np.__version__,
                         machine=platform.machine(), system=platform.system(),
                         rust_host=command("rustc", "-vV")),
        corpus=dict(functions=len(records), train_functions=int(train.sum()),
                    test_functions=int((~train).sum()), train_templates=[0, 1, 2, 3],
                    test_templates=[4, 5, 6, 7], train_opt_level=1, test_opt_level=3,
                    duplicate_codes_across_split=0,
                    native_reference_checks=len(records) * 65536,
                    unique_code_counts={split: len({r["sha256"] for r in records if r["split"] == split})
                                        for split in ("train", "test")}),
        model=dict(embedding="byte histogram + hashed adjacent-byte histogram", dimensions=1280,
                   regression="centered kernel ridge", alpha=ALPHA, test_tuning=False),
        primary=classifier_metrics(predictions, labels[~train], test_templates),
        length_only_control=classifier_metrics(length_predictions, labels[~train], test_templates),
        shuffled_training_label_control=dict(repetitions=len(negative), mean_accuracy=float(np.mean(negative)),
                                             accuracy_95_percent_range=np.quantile(negative, [0.025, 0.975]).tolist()),
        exploratory_table_decoder=table_metrics(decoded, tables[~train]),
        exploratory_length_table_decoder=table_metrics(length_decoded, tables[~train]),
        saved_tensor_predictor_reload_identical=True,
        limitations=["Only four arithmetic behaviours and one compiler/architecture/OS.",
                     "Fixed embedding; regression weights are learned, not machine semantics.",
                     "Opcode/length signatures can explain classification success.",
                     "Finite table transfer does not test unseen numeric inputs or arbitrary execution.",
                     "Four held-out template blocks limit population-level inference.",
                     "Template and optimization changes are not isolated.",
                     "Native reference checks validate the corpus, not independent learned successes."])
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("primary", "length_only_control",
                                                "shuffled_training_label_control",
                                                "exploratory_table_decoder")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
