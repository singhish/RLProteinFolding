from typing import List
import numpy as np
import random

from protein import ProteinState


class ReplayBuffer:
    def __init__(self, max_size: int):
        self.max_size: int = max_size

        self.states: List[ProteinState] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.next_states: List[ProteinState] = []

        self.size: int = 0

    def append(self,
               state: ProteinState,
               action: np.ndarray,
               reward: float,
               next_state: ProteinState):

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)

        self.size += 1
        if self.size > self.max_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)

            self.size -= 1

    def sample(self, batch_size: int) -> List[np.ndarray]:
        sample_size = min(batch_size, self.size)

        indices = random.sample(range(self.size), sample_size)
        return [
            np.array([self.states[i].angles().flatten() for i in indices]),
            np.array([self.actions[i].flatten() for i in indices]),
            np.array([self.rewards[i] for i in indices]),
            np.array([self.next_states[i].angles().flatten() for i in indices])
        ]
