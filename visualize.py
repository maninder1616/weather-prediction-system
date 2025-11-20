from __future__ import annotations

from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory(states: Sequence[str], title: str = "Weather Trajectory") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 2.2))
    # Optimized: use dict.fromkeys to preserve order and get unique states efficiently
    unique = list(dict.fromkeys(states))
    mapping = {s: i for i, s in enumerate(unique)}
    
    # For large datasets, sample points for visualization to improve performance
    n = len(states)
    if n > 10000:
        # Sample every nth point for very large trajectories
        step = max(1, n // 10000)
        sampled_states = states[::step]
        sampled_indices = np.arange(0, n, step)
        y = [mapping[s] for s in sampled_states]
        ax.step(sampled_indices, y, where="post")
    else:
        y = [mapping[s] for s in states]
        ax.step(range(n), y, where="post")
    
    ax.set_yticks(list(mapping.values()), list(mapping.keys()))
    ax.set_xlabel("Day")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_distribution_over_time(
    states: Sequence[str],
    state_space: Sequence[str],
    title: str = "State Counts Over Time",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3))
    n = len(states)
    xs = np.arange(n)
    
    # Optimized: use numpy for efficient counting
    state_to_idx = {s: i for i, s in enumerate(state_space)}
    state_indices = np.array([state_to_idx.get(s, 0) for s in states])
    
    # Pre-allocate arrays for cumulative counts
    cumulative_counts = np.zeros((len(state_space), n), dtype=np.float64)
    
    for i, s in enumerate(state_space):
        # Use vectorized operations for better performance
        mask = (state_indices == state_to_idx[s])
        cumulative = np.cumsum(mask, dtype=np.float64)
        cumulative_counts[i] = cumulative / np.arange(1, n + 1, dtype=np.float64)
        ax.plot(xs, cumulative_counts[i], label=s)
    
    ax.set_xlabel("Day")
    ax.set_ylabel("Empirical Probability")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_transition_heatmap(
    transition_matrix: np.ndarray,
    state_space: Sequence[str],
    title: str = "Transition Probabilities",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.8, 4))
    im = ax.imshow(transition_matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(state_space)), state_space, rotation=45, ha="right")
    ax.set_yticks(range(len(state_space)), state_space)
    ax.set_xlabel("Next State")
    ax.set_ylabel("Current State")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, label="P(next|current)")
    fig.tight_layout()
    return fig





