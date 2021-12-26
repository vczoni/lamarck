from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ParallelMode(Enum):
    OFF = 0
    MULTITHREAD = 2
    MULTIPROCESS = 3


@dataclass
class SimulationConfig:
    """
    Simulation configurations.
    """
    max_generations: int = 20
    max_stall: int = 5
    p_selection: float = 0.5
    p_selection_weak: float = 0.
    randomize_to_fill_pop: bool = False
    n_dispute: int = 2
    n_parents: int = 2
    children_per_relation: int = 2
    p_mutation: float = 0.05
    max_mutated_genes: int = 1
    children_per_mutation: int = 1
    p_new_random: float = 0.
    parallel_processing: int = ParallelMode.OFF
    max_workers: int | None = None
    peek_champion_variables: list | None = None

    def set_parallel_processing(self, mode: int = 0) -> None:
        """
        Modes
        -----
            0: OFF
            1: MULTITHREAD
            2: MULTIPROCESS
        """
        selected_mode = {
            0: ParallelMode.OFF,
            1: ParallelMode.MULTITHREAD,
            2: ParallelMode.MULTIPROCESS
        }[mode]
        self.parallel_processing = selected_mode
