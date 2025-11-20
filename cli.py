from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from probabilities_in_the_sky.models.markov_chain import MarkovChain
from probabilities_in_the_sky.simulation.simulator import simulate_markov_chain
from probabilities_in_the_sky.viz.visualize import (
    plot_trajectory,
    plot_distribution_over_time,
    plot_transition_heatmap,
)


def parse_matrix(matrix_str: str) -> np.ndarray:
    rows = matrix_str.strip().split(";")
    matrix = []
    for r in rows:
        matrix.append([float(x) for x in r.split(",")])
    return np.array(matrix, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Markov Chain Weather Simulator")
    parser.add_argument("--states", nargs="+", default=["Sunny", "Cloudy", "Rainy"], help="List of states")
    parser.add_argument(
        "--matrix",
        type=str,
        default="0.7,0.2,0.1;0.3,0.5,0.2;0.2,0.3,0.5",
        help="Row-stochastic matrix string, rows separated by ';'",
    )
    parser.add_argument("--initial", type=str, default="Sunny", help="Initial state")
    parser.add_argument("--days", type=int, default=30, help="Number of days to simulate")
    parser.add_argument("--plot", type=str, default="", help="Optional path to save a composite plot")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    states = args.states
    matrix = parse_matrix(args.matrix)
    mc = MarkovChain(states, matrix)
    trajectory, counts = simulate_markov_chain(mc, args.initial, args.days, seed=args.seed)

    print("Trajectory:", ", ".join(trajectory))
    print("Counts:", counts)

    # Create a composite figure layout
    fig_traj = plot_trajectory(trajectory)
    fig_dist = plot_distribution_over_time(trajectory, mc.states)
    fig_heat = plot_transition_heatmap(mc.transition_matrix, mc.states)

    if args.plot:
        out_dir = Path(args.plot).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_traj.savefig(Path(args.plot).with_stem(Path(args.plot).stem + "_trajectory"), dpi=160)
        fig_dist.savefig(Path(args.plot).with_stem(Path(args.plot).stem + "_distribution"), dpi=160)
        fig_heat.savefig(Path(args.plot).with_stem(Path(args.plot).stem + "_heatmap"), dpi=160)
        print(f"Saved plots near: {args.plot}")


if __name__ == "__main__":
    main()


