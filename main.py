<<<<<<< HEAD
import numpy as np
from .qlearner import QLearningAgent
from .qlearner.environment.custom_env import CustomEnvironment
from .qlearner.environment.base_env import Environment
import matplotlib.pyplot as plt


env = CustomEnvironment((1, 1, 'LEFT'), (4, 4), [(1,1)])
# Initialize state and action space
action_space = ['TURN RIGHT', 'TURN LEFT', 'FORWARD', 'BACKWARD']
state_space = [(x, y, o) for x in range(0, 20+1) for y in range(0, 20+1) for o in ['UP','RIGHT','LEFT','DOWN']]
goals = [(1,1,'DOWN')]
#print(state_space)


qlearner = QLearningAgent(state_space=state_space, action_space=action_space, env=env, lr=.01, discount=.97,explore=.5 , terminals=[(10,10,'UP')], debug=False, n_eps=20, n_steps=10000)
qlearner.train(load_qtable=True, save_qtable=True)
print(qlearner.rew_per_ep)
print(qlearner.goals_found)

plt.plot(qlearner.rew_per_ep)
plt.show()
=======
import numpy as np
from .qlearner import QLearningAgent
from .custom_env import CustomEnvironment
from .base_env import Environment
import matplotlib.pyplot as plt


env = CustomEnvironment((1, 1, 'LEFT'), (4, 4), [(1,1)])
# Initialize state and action space
action_space = ['TURN RIGHT', 'TURN LEFT', 'FORWARD', 'BACKWARD']
state_space = [(x, y, o) for x in range(0, 20+1) for y in range(0, 20+1) for o in ['UP','RIGHT','LEFT','DOWN']]
goals = [(1,1,'DOWN')]
#print(state_space)


qlearner = QLearningAgent(state_space=state_space, action_space=action_space, env=env, lr=.1, discount=1, terminals=[(10,2,'UP')], debug=False, n_eps=200)
qlearner.train()
print(qlearner.rew_per_ep)
print(qlearner.goals_found)

plt.plot(qlearner.rew_per_ep)
plt.show()
>>>>>>> 65a689bcbb37f1ce53129abf47860072e9c092ec
