"""Models the grid world environment for fully observable environemnt."""

import sys
import numpy as np
import time
from .globals import * 
import numpy as np
from .base_env import Environment


class FullyObservedEnvironmentModel(Environment):
    """
    Models the grid world environment for fully observable environemnt.
    """

    def __init__(self, state, goal, obstacles):
        super().__init__()
        self.curr_state = state # snapshot of env
        self.t = 0 # time that increments by timestep
        self.goal_state = goal
        self.obstacles = obstacles
    
    def state_transition(self, state, action):
        """
        Transitions the environment to a new state given actions.

        Args:
            state (state): snapshot of environment
            action (action): action chosen by agent

        Returns:
            next_state: new snapshot of environment
        """

        x, y, orien = state

        if action == 'TURN RIGHT':
            if orien == 'UP':
                orien = 'RIGHT'
            elif orien == 'RIGHT':
                orien = 'DOWN'
            elif orien == 'DOWN':
                orien = 'LEFT'
            elif orien == 'LEFT':
                orien = 'UP'
            
        elif action == 'TURN LEFT':
            if orien == 'UP':
                orien = 'LEFT'
            elif orien == 'LEFT':
                orien = 'DOWN'
            elif orien == 'DOWN':
                orien = 'RIGHT'
            elif orien == 'RIGHT':
                orien = 'UP'
            
        elif action == 'FORWARD':
            if orien == 'UP':
                y = y + 1
            elif orien == 'DOWN':
                y = y - 1
            elif orien == 'RIGHT':
                x = x + 1
            elif orien == 'LEFT':
                x = x - 1

        elif action == 'BACKWARD':
            if orien == 'UP':
                y = y - 1
            elif orien == 'DOWN':
                y = y + 1
            elif orien == 'RIGHT':
                x = x - 1
            elif orien == 'LEFT':
                x = x + 1

        new_state = (x, y, orien)
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
    
    def print_state(self, state, action, next_state, reward, ep, it):
        print('-------------------------')
        print('EPISODE:', ep, 'ITERATION', it )
        print("STATE:", state, " ACTION: ",action)
        print("NEXT STATE: ", next_state, " REWARD: ", reward)
        print('-------------------------')

    def reward_func(self, state, next_state):
        """
        Returns the reward for being in the new state, and also performs state validation

        Args:
            state (tuple): prev state
            next_state (tuple): next state

        Returns:
            float, tuple: reward, validated state
        """
        
        x_1, y_1, orien_1 = state
        x_2, y_2, orien_2 = next_state

        reward = 0
        
        if (x_2, y_2)in self.obstacles:
            reward = OBSTACLE_REWARD
            x_2 = x_1
            y_2 = y_1
            orien_2 = orien_1

        if (x_2,y_2) == self.goal_state:
            reward = GOAL_REWARD

        if x_2 > X_DIRECTION or y_2 > Y_DIRECTION or x_2 < 1 or  y_2 < 1:
            x_2 = x_1
            y_2 = y_1
            orien_2 = orien_1
            reward = OBSTACLE_REWARD
           
        next_state = (x_2, y_2, orien_2)
        return reward, next_state
    
    
    def reset(self):
        return (1,1,'UP')

    def is_terminal(self, state, terminals):
        state_np = np.array(state)
        return np.any(np.all(terminals == state_np, axis=1))
