import cvxpy as cp
import torch
import numpy as np
from cvxpylayers.torch import CvxpyLayer

X_LOWER_BOUND = -10
X_UPPER_BOUND = 10
TRAIN_SLOPE = 2  

class BurdenDELTA:
    def __init__(self, x_dim, funcs):
        self.cost = 1/x_dim
        self.x = cp.Variable(x_dim)
        self.r = cp.Parameter(x_dim)
        self.w = cp.Parameter(x_dim)
        self.b = cp.Parameter(1)
        self.f_der = cp.Parameter(x_dim)

        target = (
            self.x @ self.f_der
            - self.g(self.x, self.w, self.b, TRAIN_SLOPE)
            - self.c(self.x, self.r)
        )
        constraints = [
            self.x >= X_LOWER_BOUND,
            self.x <= X_UPPER_BOUND
        ]
        problem = cp.Problem(cp.Maximize(target), constraints)
        self.layer = CvxpyLayer(problem, parameters=[self.r, self.w, self.b, self.f_der], variables=[self.x])

    def optimize_X(self, X, w, b, F_DER):
        return self.layer(X, w, b, F_DER)[0]

    def score(self, x, w, b):
        return x@w + b

    def g(self, x, w, b, slope):
        return 0.5*cp.norm(cp.hstack([1, (slope*self.score(x, w, b) - 1)]), 2)

    def c(self, x, r):
        return self.cost*cp.sum_squares(x-r)
