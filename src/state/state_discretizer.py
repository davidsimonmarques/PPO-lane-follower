"""Discretizador de observações para estados tabulares."""

import numpy as np
from typing import Dict, Sequence

class StateDiscretizer:
    def __init__(self, bins: Dict[str, Sequence[float]]):
        self.bins = bins

    def discretize(self, observation: Dict) -> int:
        """Converte observações contínuas em um índice de estado discreto."""
        values = []
        for key, bin_edges in self.bins.items():
            value = observation.get(key, 0.0)
            values.append(np.digitize(value, bin_edges))
        # TODO: usar codificação composta ou hashing simples
        index = 0
        for value in values:
            index = index * (len(bin_edges) + 1) + value
        return int(index)

    def state_space_size(self) -> int:
        total = 1
        for edges in self.bins.values():
            total *= (len(edges) + 1)
        return total
