import numpy as np
from typing import List, Optional, Union, Any, Callable
from tabulate import tabulate

class TemporalDifferenceLearner():


    def __init__(self, name:str, state_space:List[Any], action_space:List[Any], env:Any, terminals:List[Any], lr:float, discount: float, debug:bool):
        
        self.name = name
        self.state_space = state_space
        self.action_space = action_space
        self.lr = lr
        self.discount = discount
        self.env = env
        self.terminals = np.array(terminals)
        self.debug = debug
        
    def policy(self):
        pass
        
    def run_episode(self):
        pass

    def run_step(self):
        pass

    def train(self):
        pass

