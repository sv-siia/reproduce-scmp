import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

X_LOWER_BOUND = -10
X_UPPER_BOUND = 10

class BURDEN():
    def __init__(self, x_dim, funcs):
        self.c = funcs["c"]
        self.score = funcs["score"]

        self.x = cp.Variable(x_dim)
        self.r = cp.Parameter(x_dim)
        self.w = cp.Parameter(x_dim)
        self.b = cp.Parameter(1)

        target = self.c(self.x, self.r)
        constraints = [
            self.score(self.x, self.w, self.b) >= 0,
            self.x >= X_LOWER_BOUND,
            self.x <= X_UPPER_BOUND
        ]

        objective = cp.Minimize(target)
        problem = cp.Problem(objective, constraints)
        self.layer = CvxpyLayer(problem, parameters=[self.r, self.w, self.b], variables=[self.x])

    def calc_burden(self, X, Y, w, b):
        Xpos = X[Y == 1]
        if len(Xpos) == 0:
            return 0
        Xmin = self.layer(Xpos, w, b)[0]
        return torch.mean(torch.sum((Xpos - Xmin) ** 2, dim=1))
