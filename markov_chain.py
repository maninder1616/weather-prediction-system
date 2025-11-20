from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Sequence

import numpy as np


@dataclass(frozen=True)
class MarkovChain:
    """Discrete-time Markov Chain with named states and a row-stochastic matrix."""

    states: List[str]
    transition_matrix: np.ndarray

    def __init__(self, states: Sequence[str], transition_matrix: Sequence[Sequence[float]]):
        states_list = list(states)
        matrix = np.array(transition_matrix, dtype=float)

        if matrix.ndim != 2:
            raise ValueError("transition_matrix must be 2D")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("transition_matrix must be square")
        if len(states_list) != matrix.shape[0]:
            raise ValueError("states length must match matrix size")

        # Validate non-negativity and row-stochastic property
        if (matrix < -1e-12).any():
            raise ValueError("transition_matrix contains negative probabilities")
        row_sums = matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-8):
            raise ValueError("each row in transition_matrix must sum to 1")

        object.__setattr__(self, "states", states_list)
        object.__setattr__(self, "transition_matrix", matrix)

    @property
    def num_states(self) -> int:
        return len(self.states)

    def state_index(self, state: str) -> int:
        try:
            return self.states.index(state)
        except ValueError as e:
            raise KeyError(f"Unknown state: {state}") from e

    def next_state_distribution(self, current_state_index: int) -> np.ndarray:
        return self.transition_matrix[current_state_index]

    def step(self, current_state_index: int, rng: np.random.Generator | None = None) -> int:
        rng = rng or np.random.default_rng()
        probs = self.next_state_distribution(current_state_index)
        return int(rng.choice(self.num_states, p=probs))

    def stationary_distribution(self) -> np.ndarray:
        """Compute stationary distribution via eigen-decomposition (for educational purposes)."""
        eigvals, eigvecs = np.linalg.eig(self.transition_matrix.T)
        idx = np.argmin(np.abs(eigvals - 1))
        stationary = np.real(eigvecs[:, idx])
        stationary /= stationary.sum()
        stationary = np.where(stationary < 0, 0, stationary)
        stationary /= stationary.sum()
        return stationary


