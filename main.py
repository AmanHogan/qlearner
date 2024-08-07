import numpy as np
from .learning.learners.qlearner import QLearningAgent
from .learning.environment.custom_env import GridWorldExample
import matplotlib.pyplot as plt

env = GridWorldExample(start=(1,1), goal=(8,8), obstacles=[(4,3)], max_x=20, max_y=20)

qlearner = QLearningAgent(state_space=env.state_space, action_space=env.action_space, env=env, lr=.001, discount=.998, explore=.3, terminals=[env.goal_state], debug=False, n_eps=100, n_steps=5000, file_path='./qlearner/q_table')
qlearner.train(load_qtable=True, save_qtable=True)

print(qlearner.get_learner_info())

plt.plot(qlearner.avg_rewards_per_ep)
plt.show()

print(qlearner)

