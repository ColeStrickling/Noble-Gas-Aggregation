#!/usr/bin/env python3
"""Simple PMF plotting helper."""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--xlabel", default="r (Å)")
    parser.add_argument("--ylabel", default="W (kJ/mol)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if len(args.input) != len(args.labels):
        raise ValueError("Number of input files must match number of labels.")

    plt.figure()
    for filename, label in zip(args.input, args.labels):
        data = np.loadtxt(filename, comments="#")
        plt.plot(data[:, 0], data[:, 1], label=label)

    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)


if __name__ == "__main__":
    main()
