# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as nr


class OUNoise:
    """
        ORNSTEIN UHLENBECK Noise
    """

    def __init__(self, action_dimension, mean, sigma, mu=0, theta=0.15):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.mean = mean
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        # theta * (mu - x) + N(mean,sigma)
        dx = self.theta * (self.mu - x) + self.mean + \
            np.multiply(self.sigma, nr.randn(len(x)))
        self.state = x + dx
        return self.state


if __name__ == "__main__":
    # 각 픽셀의 rgb 체널에 노이즈를 준다.
    ou = OUNoise(3)     # state dimension is 3 (height, width, channels)
    states = []
    for i in range(1000):
        states.append(ou.noise())

    import matplotlib.pyplot as plt
    plt.plot(states)
    plt.show()
