import numpy as np
from typing import List, Optional, Union, Any, Callable

class QLearningAgent:

    DEFAULT_LR = .01
    DEFAULT_DISCOUNT = 1
    DEAULT_EXPLORE = .5
    DEFAULT_NSTEPS = 100
    DEFAULT_GOAL_MSG = "Reached Terminal state"

    def __init__(self, state_space: List[Any], action_space: List[Any], 
                 env: Any, terminals: List[Any]=[], 
                 policy: Optional [Union[Callable[[Any], Any], None]] = None, 
                 lr: float = DEFAULT_LR, discount: float = DEFAULT_DISCOUNT, 
                 explore: float = DEAULT_EXPLORE, terminal_msg: str = DEFAULT_GOAL_MSG):
                
        self.state_space = state_space
        self.action_space = action_space
        self.q_table = None
                
        self.lr = lr
        self.discount = discount
        self.explore = explore
        self.exploit = 1 - self.explore
    
        self.terminals = np.array(terminals)
        self.env = env

        if policy != None:
            self.policy = policy
        else:
            self.policy = self.greedy

        self.goals_found = 0
        self.total_rewards = 0

        self.rewards_per_episode = np.array([])
        self.history = np.empty((0, 6), dtype=object)  

        self.episode_count = 0
        self.terminal_msg = terminal_msg

    def init_q_table(self) -> None:
        self.q_table = np.zeros((len(self.state_space), len(self.action_space)))
    
    def stoi(self, state: Any) -> int:
        """Gets index of state in statespace

        Args:
            state (Any): state

        Returns:
            int: state index
        """
        return self.state_space.index(state)

    def atoi(self, action: Any) -> int:
        """Gets index of action in actions space

        Args:
            action (Any): action

        Returns:
            int: action index
        """
        return self.action_space.index(action)
    
    def run_episode(self, max_steps: int = DEFAULT_NSTEPS, debug:bool = True) -> None:
        """Runs a full episode given the number of steps

        Args:
            max_steps (int, optional): number of steps. Defaults to DEFAULT_NSTEPS.
            debug (bool, optional): prints out info for each step. Defaults to True.
        """

        state = self.env.reset() 
        total_rewards = 0

        for step in range(max_steps):

            action = self.policy(state)
            next_state, reward = self.env.step(state, action)
            
            self.update_table(state, action, next_state, reward)
            self.update_info(state, action, next_state, reward, self.episode_count,step )
            
            if debug:
                self.print_info()
            
            total_rewards += reward

            if self.env.is_terminal(state, self.terminals):
                print(self.terminal_msg )
                self.goals_found += 1
                break
            
            
            state = next_state

        self.episode_count += 1
        self.update_rewards_per_epsiode(total_rewards)
    
    def greedy(self, state) -> Any:

        if np.random.choice(['explore', 'exploit'], p=[self.explore, self.exploit]) == 'explore':
            return np.random.choice(self.action_space)
        else:
            s_idx = self.stoi(state)
            a_idx = np.argmax(self.q_table[s_idx])
            return self.action_space[a_idx]

    def update_table(self, state:Any, action:Any, next_state:Any, reward:Any) -> None:
        s_idx = self.stoi(state)
        a_idx = self.atoi(action)
        ns_idx = self.stoi(next_state)
        self.q_table[s_idx, a_idx] += self.lr * (reward + self.discount * np.max(self.q_table[ns_idx]) - self.q_table[s_idx, a_idx])

    def update_rewards_per_epsiode(self, total_rewards):
        self.rewards_per_episode = np.append(self.rewards_per_episode, total_rewards)
        
    def update_info(self, s, a, ns, r, ep, step):
        self.history = np.append(self.history,  np.array([[s, a, ns, r, ep, step]], dtype=object), axis=0)

    def print_info(self):
        print(self.history[len(self.history)-1])
