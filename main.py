import numpy as np
from .qlearner import QLearningAgent
from .custom_env import CustomEnvironment
from .base_env import Environment


env = CustomEnvironment((1, 1, 'LEFT'), (4, 4), [(1,1)])
# Initialize state and action space
action_space = ['TURN RIGHT', 'TURN LEFT', 'FORWARD', 'BACKWARD']
state_space = [(x, y, o) for x in range(0, 20+1) for y in range(0, 20+1) for o in ['UP','RIGHT','LEFT','DOWN']]
goals = [(1,1,'DOWN')]
#print(state_space)


neps = 50
nsteps = 100

qlearner = QLearningAgent(state_space=state_space, action_space=action_space, env=env, lr=.01, discount=1, terminals=[(10,2,'UP')])
qlearner.init_q_table()

for episode in range(neps):
    
    s = qlearner.env.reset() # intialize s
    tr = 0

    for step in range(nsteps): 

        a = qlearner.policy(s) # choose a from policy
        ns, r = qlearner.env.step(s, a) # take a and observe r, s'
        tr += r # inc total rewards

        qlearner.update_table(s,a,ns,r) # bellman update
        qlearner.update_info(s,a,ns,r,episode,step) # update learner
        
        # terminal check
        if qlearner.env.is_terminal(s, qlearner.terminals):
            print(qlearner.terminal_msg)
            qlearner.goals_found += 1
            break
        
        s = ns # update state

    qlearner.update_rewards_per_epsiode(tr) # update rewards
print(qlearner.goals_found)

qlearner = QLearningAgent(state_space=state_space, action_space=action_space, env=env, lr=.01, discount=1, terminals=[(10,2,'UP')])
qlearner.init_q_table()
for episode in range(neps):
    qlearner.run_episode(debug=False)
qlearner.print_info()
print(qlearner.goals_found)
