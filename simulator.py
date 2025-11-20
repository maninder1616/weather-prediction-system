from __future__ import annotations

from typing import List, Tuple, Dict

import numpy as np

from probabilities_in_the_sky.models.markov_chain import MarkovChain


def simulate_markov_chain(
    mc: MarkovChain,
    initial_state: str | int,
    days: int,
    seed: int | None = None,
) -> Tuple[List[str], Dict[str, int]]:
    if days < 0:
        raise ValueError("days must be >= 0")

    rng = np.random.default_rng(seed)

    if isinstance(initial_state, int):
        current_idx = int(initial_state)
    else:
        current_idx = mc.state_index(initial_state)

    
    trajectory_indices = np.empty(days + 1, dtype=int)
    trajectory_indices[0] = current_idx
    
    
    for i in range(1, days + 1):
        current_idx = mc.step(current_idx, rng)
        trajectory_indices[i] = current_idx

    
    trajectory_states = [mc.states[i] for i in trajectory_indices]
    
   
    counts: Dict[str, int] = {}
    unique, counts_arr = np.unique(trajectory_indices, return_counts=True)
    for idx, count in zip(unique, counts_arr):
        counts[mc.states[idx]] = int(count)
    
   
    for state in mc.states:
        if state not in counts:
            counts[state] = 0
    
    return trajectory_states, counts




