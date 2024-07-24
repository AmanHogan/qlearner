"""Models the grid world environment for fully observable environemnt."""

import sys
import numpy as np
import time
import numpy as np
from .base_env import Environment


class GridWorldExample (Environment):
    """
    Models the grid world environment for fully observable environemnt.
    """

    def __init__(self, start, goal, obstacles, max_x:int = 20, max_y:int = 20):
        super().__init__()
        self.curr_state = start
        self.t = 0
        self.goal_state = goal
        self.obstacles = obstacles
        self.max_x = max_x
        self.max_y = max_y
        self.state_space = [(x, y) for x in range(1, max_x+1) for y in range(1, max_y+1)]
        self.action_space = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    
    def state_transition(self, state, action):
        """
        Transitions the environment to a new state given actions.

        Args:
            state (state): snapshot of environment
            action (action): action chosen by agent

        Returns:
            next_state: new snapshot of environment
        """

        x, y = state

        if action == 'UP':
            y = y + 1
        elif action == 'DOWN':
            y = y - 1
        elif action == 'LEFT':
            x = x - 1
        elif action == 'RIGHT':
            x = x + 1

        new_state = (x, y)

        return new_state

    def step(self, state, action):
        """
        Gets a new state and reward based on state\n
        Args:
            state (state): snapshot of environment
            action (action): action chosen by agent
        Returns:
            new_state, reward: the new state of envirnoment and the reward for being in that state
        """

        next_state_ = self.state_transition(state, action)
        next_reward, next_state = self.reward_func(state, next_state_)
        self.t = self.t + 1

        return next_state, next_reward
    

    def reward_func(self, state, next_state):
        """
        Returns the reward for being in the new state, and also performs state validation

        Args:
            state (tuple): prev state
            next_state (tuple): next state

        Returns:
            float, tuple: reward, validated state
        """
        
        x_1, y_1 = state
        x_2, y_2 = next_state

        reward = 0
        
        if (x_2, y_2) in self.obstacles:
            reward += -.1
            x_2 = x_1
            y_2 = y_1

        if (x_2,y_2) == self.goal_state:
            reward = 100

        if x_2 > self.max_x or y_2 > self.max_y or x_2 < 1 or  y_2 < 1:
            x_2 = x_1
            y_2 = y_1
            reward += -.1
           
        else:
            reward = -0
        
        next_state = (x_2, y_2)
        return reward, next_state
    
    
    def reset(self):
        return (1,1)

    def is_terminal(self, state, terminals):
        state_np = np.array(state)
        return np.any(np.all(terminals == state_np, axis=1))
