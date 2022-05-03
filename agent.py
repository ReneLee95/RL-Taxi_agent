import numpy as np
from collections import defaultdict
import sys


def select_action(Q,state):
    #action = np.random.choice(self.n_actions)
    if np.random.rand(0,6) < 0.01:
        return np.random.rand(0,6)
    else:
        return np.argmax(Q[state])

class Agent:

    def __init__(self, Q, mode="test_mode"):
        self.Q = Q
        self.mode = mode
        self.n_actions = 6
        self.alpha = 0.0
        self.gamma = 0.0
        self.epsilon = 0.01
        if mode == "mc_control":
            self.AllEpisode = list()

    def select_action(self, state):
        #action = np.random.choice(self.n_actions)
        if np.random.rand(0,6) < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.Q[state])
        
        
    def step(self, state, action, reward, next_state, done):
        sys.path.append("taxi")
        import taxi as t
        newQ = defaultdict(lambda: np.zeros(6))
        inaction = select_action(self.Q,state)
    #    print(inaction)
        inputstep,inputreward,done,info = t.env.step(inaction)
        #print(inputstep,inputreward,done,state,action,reward,done)
        if self.mode == "q_learning":
            self.alpha = 0.1
            self.gamma = 0.7
            self.Q[state][action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])
        if self.mode == "mc_control":
            self.alpha = 0.001
            self.gamma = 0.9999
            #self.AllEpisode.append((state,action,reward))
            self.AllEpisode.append((state,inaction,inputreward))
            if done == True:
                Returns = defaultdict(lambda: np.zeros(self.n_actions))
                
                for state,action,reward in reversed(self.AllEpisode):
                    Returns[state][action] = reward + self.gamma * Returns[state][action]
                    self.Q[state][action] += self.alpha * (np.average(Returns[state][action]) - self.Q[state][action])
                    
                self.AllEpisode.clear()

