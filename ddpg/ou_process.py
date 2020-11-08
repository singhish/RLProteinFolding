import numpy as np

class OrnsteinUhlenbeckProcess:
    def __init__(self, action_dim: int, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dX = self.theta * (self.mu - self.X) + self.sigma * np.random.randn(self.action_dim)
        self.X += dX
        return self.X
