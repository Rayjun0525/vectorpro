"""Run the saved finite-table decoder using function bytes and two 4-bit inputs.

This is a learned table predictor for the four trained behaviours, not a CPU
emulator or vectorpro executable. No arithmetic class label is an input.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from run import OUT, embedding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", required=True, type=Path, help="raw .text.f bytes, not an ELF file")
    parser.add_argument("--model", type=Path, default=OUT / "model_and_predictions.npz")
    parser.add_argument("--a", required=True, type=int, choices=range(16))
    parser.add_argument("--b", required=True, type=int, choices=range(16))
    args = parser.parse_args()
    vector = embedding(args.code.read_bytes())
    with np.load(args.model) as model:
        scores = ((vector - model["embedding_mean"]) @ model["training_embedding_basis"].T
                  @ model["output_bit_dual_weights"] + model["output_bit_mean"])
    bits = scores.reshape(256, 8)[args.a * 16 + args.b] > 0
    result = int(np.sum(bits * (1 << np.arange(8))))
    print(json.dumps(dict(a=args.a, b=args.b, predicted_output=result,
                         scope="finite learned table; no guarantee for arbitrary binary code")))


if __name__ == "__main__":
    main()
