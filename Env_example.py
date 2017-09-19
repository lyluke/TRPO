import numpy as np

#Define the input dimension here
dim = 2

#Define your function here. It must support vector operation in numpy
def f(x):
    m = 20
    d = x.shape[1]
    s = np.zeros((x.shape[0], ))
    
    for i in range(d):
        xi = x[:, i]
        xi[xi < 0] = 0
        xi[xi > np.pi] = np.pi

        s += np.sin(xi)*(np.sin((i+4)*(xi**2)/np.pi))**m
    return -s

class Env:
    def __init__(
        self,
        seed):
        self.seed = seed
        np.random.seed(self.seed)
    def spec(self):
        return {'size':(dim,)}
    def reset(self, batch_size):
        #Define the starting point of each iteration. At the beginning of each iteration, agent's position will be reset
        self._state = np.pi*0.5*np.ones((batch_size,dim))
        observation = np.copy(self._state)
        return observation

    def step(self, action):
        x_old = self._state
        self._state = self._state + action
        x = self._state
        reward = f(x_old) - f(x)      
        done = False
        next_observation = np.copy(self._state)
        return next_observation, reward, done

    def render(self):
        print 'current state:', self._state
        
    def function(self, x):
        return f(np.array([x]))