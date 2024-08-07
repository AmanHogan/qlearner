import numpy as np
import logging
from typing import List, Optional, Union, Any, Callable
from tabulate import tabulate
import time
from .tdlearner import TemporalDifferenceLearner

logging.basicConfig(level=logging.INFO)

class QLearningAgent(TemporalDifferenceLearner):
    
    DEFAULT_GOAL_MSG = "Reached Terminal state"

    def __init__(self, state_space: List[Any], action_space: List[Any], env: Any, terminals: List[Any] = [], 
                 policy: Optional[Union[Callable[[Any], Any], None]] = None, lr: float = .001, 
                 discount: float = .998, explore: float = .2, 
                 terminal_msg: str = DEFAULT_GOAL_MSG, n_eps: int = 100, 
                 n_steps: int = 1000, debug: bool = False, file_path:str='q_table'):
        
        super().__init__("Q Learner", state_space, action_space, env, terminals, lr, discount, debug)
        
        self.q_table = None
        self.explore = explore
        self.exploit = 1 - explore
        self.policy = policy if policy else self.greedy
        self.goals_found = 0
        self.total_rewards = 0
        self.rew_per_ep = np.array([])
        self.avg_rewards_per_ep = np.array([])
        self.history = np.empty((0, 6), dtype=object)
        self.ep_idx = 0
        self.terminal_msg = terminal_msg
        self.n_steps = n_steps
        self.n_eps = n_eps
        self.file_path = file_path
        self.start_time = time.time()
        
        self.validate_parameters()

    def validate_parameters(self):
        
        if not isinstance(self.state_space, list) or not isinstance(self.action_space, list):
            raise ValueError("State space and action space must be lists.")
        
        if not isinstance(self.terminals, np.ndarray):
            raise ValueError("Your teminal condition states must be a list")
        
        if not (0 <= self.explore <= 1):
            raise ValueError("Explore rate must be between 0 and 1.")
        
        if not (0 <= self.lr <= 1):
            raise ValueError("Learning rate must be between 0 and 1.")
        
        if not (0 <= self.discount <= 1):
            raise ValueError("Discount factor must be between 0 and 1.")

    def init_q_table(self) -> None:
        try:
            self.q_table = np.zeros((len(self.state_space), len(self.action_space)))
            if self.debug:
                logging.info('Q table successfully created')
        except Exception as err:
            logging.error(f'Failed to initialize Q table: {err}')
            raise

    def get_state_idx(self, state: Any) -> int:
        return self.state_space.index(state)

    def get_action_idx(self, action: Any) -> int:
        return self.action_space.index(action)

    def run_episode(self, episode:int) -> None:
        s = self.env.reset()
        tr = 0

        for step in range(self.n_steps):
            
            a = self.policy(s)
            ns, reward = self.env.step(s, a)
            self.update_table(s, a, ns, reward)
            self.update_info(s, a, ns, reward, self.ep_idx, step)

            if self.debug:
                self.get_current()

            tr += reward

            if self.env.is_terminal(s, self.terminals):
                logging.info(self.terminal_msg + f' [Episode #] {episode}')
                self.goals_found += 1
                break

            s = ns

        self.ep_idx += 1
        self.rew_per_ep = np.append(self.rew_per_ep, tr)
        self.avg_rewards_per_ep = np.append(self.avg_rewards_per_ep, sum(self.rew_per_ep)/(self.ep_idx+1))

    def greedy(self, state: Any) -> Any:
        
        if self.q_table is None:
            logging.error('Q table must be initialized first!')
            raise ValueError('Q table must be initialized first!')

        if np.random.choice(['explore', 'exploit'], p=[self.explore, self.exploit]) == 'explore':
            return np.random.choice(self.action_space)
        
        else:
            s_idx = self.get_state_idx(state)
            a_idx = np.argmax(self.q_table[s_idx])
            return self.action_space[a_idx]

    def update_table(self, state: Any, action: Any, next_state: Any, reward: Any) -> None:
        
        if self.q_table is None:
            logging.error('Q table must be initialized first!')
            raise ValueError('Q table must be initialized first!')

        s_idx = self.get_state_idx(state)
        a_idx = self.get_action_idx(action)
        ns_idx = self.get_state_idx(next_state)
        self.q_table[s_idx, a_idx] += self.lr * (reward + self.discount * np.max(self.q_table[ns_idx]) - self.q_table[s_idx, a_idx])

    def update_info(self, state: Any, action: Any, next_state: Any, reward: Any, episode: Any, step: Any) -> None:
        self.history = np.append(self.history, np.array([[state, action, next_state, reward, episode, step]], dtype=object), axis=0)

    def get_current(self) -> None:
        logging.info(f'Current state-action pair: {self.history[len(self.history)-1]}')

    def train(self, load_qtable:bool = False, save_qtable:bool = False) -> None:
        
        if load_qtable:
            try:
                self.load_q_table()
            except FileNotFoundError as e:
                print(e)
                print("Starting learning from scratch")
                self.init_q_table()
        else:
            self.init_q_table()

        for ep in range(self.n_eps):
            self.run_episode(ep)
            logging.info(f'Finished episode {ep} with rewards of {self.rew_per_ep[ep]}')
        self.avg_rewards_per_ep = np.delete(self.avg_rewards_per_ep, 0)
        if save_qtable:
            self.save_q_table()

    def save_q_table(self) -> None:
        np.save(self.file_path, self.q_table)

    def load_q_table(self) -> None:
        self.q_table = np.load(self.file_path+'.npy')
        #exit()

    def get_learner_info(self):
        table =[
            ["STATE SPACE SIZE", len(self.state_space)],
            ["ACTION SPACE SIZE", len(self.action_space)],
            ["QTABLE SHAPE", self.q_table.shape],
            ["LEARNING RATE", self.lr],
            ["DISCOUNTED SUM", self.discount],
            ["EXPLORATION RATE", self.explore],
            ["NUMBER OF TERMINAL STATES", len(self.terminals)],
            ["NUMBER OF GOALS FOUND", self.goals_found],
            ["MAX NUMEBR OF STEPS", self.n_steps],
            ["MAX NUMBER OF EPISODES",self.n_eps ],
            ["FILE PATH OF QTABLE", self.file_path],
            ["RUNTIME (s)", time.time() - self.start_time]

        ]
        return str(tabulate(table))

    def set_time(self):
        self.start_time = time.time()

    def __str__(self):
            return (f"QLearner(name={self.name}, env={self.env}, state_space_size={len(self.state_space)}, action_space_size={len(self.action_space)}, "
                    f"lr={self.lr}, discount={self.discount}, explore={self.explore}, n_eps={self.n_eps}, n_steps={self.n_steps}, "
                    f"goals_found={self.goals_found}, total_rewards={self.total_rewards})")

    def __repr__(self):
        return (f"QLearner(name={self.name!r}, state_space={self.state_space!r}, action_space={self.action_space!r}, env={self.env!r}, "
                f"terminals={self.terminals!r}, policy={self.policy!r}, lr={self.lr!r}, discount={self.discount!r}, "
                f"explore={self.explore!r}, terminal_msg={self.terminal_msg!r}, n_eps={self.n_eps!r}, "
                f"n_steps={self.n_steps!r}, debug={self.debug!r}, file_path={self.file_path!r})")


    
