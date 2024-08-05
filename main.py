import numpy as np
from .learning.qlearner.qlearner import QLearningAgent
from .learning.qlearner.environment.custom_env import GridWorldExample
from .learning.qlearner.environment.base_env import Environment
import matplotlib.pyplot as plt


env = GridWorldExample(start=(1,1), goal=(8,8), obstacles=[(4,3)], max_x=10, max_y=10)

qlearner = QLearningAgent(state_space=env.state_space, action_space=env.action_space, env=env, lr=.01, discount=.998, explore=.4, terminals=[env.goal_state], debug=False, n_eps=100, n_steps=4000, file_path='./qlearner/q_table')
qlearner.train(load_qtable=True, save_qtable=True)

print(qlearner.get_learner_info())

plt.plot(qlearner.avg_rewards_per_ep)
plt.show()
