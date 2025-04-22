import cvxpy as cp
import torch
import numpy as np

X_LOWER_BOUND = -30
X_UPPER_BOUND = 30

class CCP:
    def __init__(self, x_dim):

        self.x = cp.Variable(x_dim)
        self.xt = cp.Parameter(x_dim)
        self.r = cp.Parameter(x_dim)
        self.w = cp.Parameter(x_dim)
        self.b = cp.Parameter(1)
        self.slope = cp.Parameter(1)

        target = self.x@self.f_derivative(self.xt, self.w, self.b, self.slope)-self.g(self.x, self.w, self.b, self.slope)-self.c(self.x, self.r, x_dim)
        constraints = [self.x >= X_LOWER_BOUND,
                       self.x <= X_UPPER_BOUND]
        self.prob = cp.Problem(cp.Maximize(target), constraints)

    
    def score(self, x, w, b):
        return x@w + b

    def f(self, x, w, b, slope):
        return 0.5*cp.norm(cp.hstack([1, (slope*self.score(x, w, b) + 1)]), 2)

    def g(self, x, w, b, slope):
        return 0.5*cp.norm(cp.hstack([1, (slope*self.score(x, w, b) - 1)]), 2)

    def f_derivative(self, x, w, b, slope):
        return 0.5*cp.multiply(slope*((slope*self.score(x, w, b) + 1)/cp.sqrt((slope*self.score(x, w, b) + 1)**2 + 1)), w)
    
    def c(self, x, r):
        return cp.sum_squares(x-r)/70

    def ccp(self, r):
        """
        numpy to numpy
        """
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

    def optimize_X(self, X, w, b, B_SPAN, slope):
        """
        tensor to tensor
        """
        X = X.numpy()
        w = w.detach().numpy()
        b = b.detach().numpy()
        slope = np.full(1, slope)

        self.w.value = w
        self.b.value = b
        self.slope.value = slope

        return torch.stack([torch.from_numpy(self.ccp(x)) for x in X])


class CCP_MANIFOLD(CCP):
    """
    Convex-Concave Problem Solver for Manifold Learning.
    Inherits from CCP and adds a manifold constraint.
    """

    def __init__(self, x_dim, h_dim, funcs):
        super().__init__(x_dim)
        
        self.x_dim = x_dim
        self.x = cp.Variable(x_dim)
        self.v = cp.Variable(h_dim)
        self.xt = cp.Parameter(x_dim)
        self.r = cp.Parameter(x_dim)
        self.w = cp.Parameter(x_dim)
        self.b = cp.Parameter(1)
        self.B_span = cp.Parameter((x_dim, h_dim))
        self.slope = cp.Parameter(1)

        target = self.x@self.f_derivative(self.xt, self.w, self.b, self.slope)-self.g(self.x, self.w, self.b, self.slope)-self.c(self.x, self.r, x_dim)
        constraints = [self.x >= X_LOWER_BOUND,
                       self.x <= X_UPPER_BOUND,
                      self.B_span@self.v == self.x-self.r]
        self.prob = cp.Problem(cp.Maximize(target), constraints)

    def ccp(self, r, B_span):
        """
        numpy to numpy
        """
        self.xt.value = r
        self.r.value = r
        self.B_span.value = B_span
        result = self.prob.solve()
        diff = np.linalg.norm(self.xt.value - self.x.value)
        cnt = 0
        while diff > 0.0001 and cnt < 10:
            cnt += 1
            self.xt.value = self.x.value
            result = self.prob.solve()
            diff = np.linalg.norm(self.x.value - self.xt.value)
        return self.x.value

    def optimize_X(self, X, w, b, B_SPAN, slope):
        """
        tensor to tensor
        """
        X = X.numpy()
        w = w.detach().numpy()
        b = b.detach().numpy()
        B_SPAN = B_SPAN.numpy()
        slope = np.full(1, slope)

        self.w.value = w
        self.b.value = b
        self.slope.value = slope

        return torch.stack([torch.from_numpy(self.ccp(x, B_span)) for x, B_span in zip(X, B_SPAN)])
