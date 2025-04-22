import cvxpy as cp
import torch
import numpy as np

X_LOWER_BOUND = -10
X_UPPER_BOUND = 10

class CCP:
    def __init__(self, x_dim, funcs):
        self.f_derivative = funcs["f_derivative"]
        self.g = funcs["g"]
        self.c = funcs["c"]

        self.x = cp.Variable(x_dim)
        self.xt = cp.Parameter(x_dim)
        self.r = cp.Parameter(x_dim)
        self.w = cp.Parameter(x_dim)
        self.b = cp.Parameter(1)
        self.slope = cp.Parameter(1)

        target = (
            self.x @ self.f_derivative(self.xt, self.w, self.b, self.slope)
            - self.g(self.x, self.w, self.b, self.slope)
            - self.c(self.x, self.r)
        )
        constraints = [
            self.x >= X_LOWER_BOUND,
            self.x <= X_UPPER_BOUND
        ]
        self.prob = cp.Problem(cp.Maximize(target), constraints)

    def ccp(self, r):
        self.xt.value = r
        self.r.value = r
        result = self.prob.solve()
        diff = np.linalg.norm(self.xt.value - self.x.value)
        cnt = 0
        while diff > 0.0001 and cnt < 10:
            cnt += 1
            self.xt.value = self.x.value
            result = self.prob.solve()
            diff = np.linalg.norm(self.x.value - self.xt.value)
        return self.x.value

    def optimize_X(self, X, w, b, slope):
        w = w.detach().numpy()
        b = b.detach().numpy()
        slope = np.full(1, slope)
        X = X.numpy()

        self.w.value = w
        self.b.value = b
        self.slope.value = slope

        return torch.stack([torch.from_numpy(self.ccp(x)) for x in X])
