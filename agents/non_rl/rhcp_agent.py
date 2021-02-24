"""
https://arxiv.org/pdf/1901.08925.pdf
heuristics-based: Recursive Handheld Cards Partitioning algorithm
The general idea of RHCP Algorithm is to 045 take a best cards handing out strategy at each step.
It decides the hand to play purely by some hand- crafted hand value estimation function: pick a partitioning strategy
with the highest Strategy Score
"""

import numpy as np

from doudizhu.utils import CARD_TYPE, INDEX


class RHCPAgent(object):
    '''
    Dou Dizhu Rule agent version 1
    '''

    def __init__(self):
        self.use_raw = True

    def step(self, state):
        '''
        Predict the action given raw state. A naive rule.
        Args:
            state (dict): Raw state from the doudizhu

        Returns:
            action (str): Predicted action
        '''
        action = 0
        return action

    def eval_step(self, state):
        '''
        Step for evaluation. The same to step
        '''
        return self.step(state), []

