import numpy as np
from .qlearner.qlearner import QLearningAgent
from .qlearner.environment.custom_env import GridWorldExample
from .qlearner.environment.base_env import Environment
import matplotlib.pyplot as plt


env = GridWorldExample(start=(1,1), goal=(4,4), obstacles=[(5,5), (8,8)], max_x=10, max_y=10)

qlearner = QLearningAgent(state_space=env.state_space, action_space=env.action_space, env=env, lr=.01, discount=.998, explore=.1, terminals=[env.goal_state], debug=False, n_eps=1000, n_steps=100, file_path='./qlearner/q_table')
qlearner.train(load_qtable=True, save_qtable=True)

print(qlearner.get_learner_info())

plt.plot(qlearner.avg_rewards_per_ep)
plt.show()
