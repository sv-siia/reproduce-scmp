import cvxpy as cp
import torch
import numpy as np
from strategic_classification.models.baseccp import BaseCCP

X_LOWER_BOUND = -10
X_UPPER_BOUND = 10

class BatchedCCP(BaseCCP):
    """Class for solving the convex-concave problem using CVXPY."""
    def __init__(self, x_dim, batch_size, scale):
        super().__init__(x_dim)
        self.batch_size = batch_size
        self.x = cp.Variable((batch_size, x_dim))
        self.xt = cp.Parameter((batch_size, x_dim))
        self.r = cp.Parameter((batch_size, x_dim))
        self.w = cp.Parameter(x_dim)
        self.b = cp.Parameter(1)
        self.slope = cp.Parameter(1)

        target = cp.diag(self.x@(self.f_derivative_batch(self.xt, self.w, self.b, self.slope).T))-self.g_batch(self.x, self.w, self.b, self.slope)-self.c_batch(self.x, self.r, x_dim, scale)
        constraints = [self.x >= X_LOWER_BOUND,
                       self.x <= X_UPPER_BOUND]
        self.prob = cp.Problem(cp.Maximize(cp.sum(target)), constraints)
        
    def ccp(self, r):
        """
        numpy to numpy
        """
        self.xt.value = r
        self.r.value = r
        result = self.prob.solve()
        diff = np.linalg.norm(self.xt.value - self.x.value)
        cnt = 0
        while diff > 0.001 and cnt < 100:
            cnt += 1
            self.xt.value = self.x.value
            result = self.prob.solve()
            diff = np.linalg.norm(self.x.value - self.xt.value)/self.batch_size
        return self.x.value
    
    def optimize_X(self, X, w, b, slope):
        """
        tensor to tensor
        """
        w = w.detach().numpy()
        b = b.detach().numpy()
        slope = np.full(1, slope)
        X = X.numpy()
        
        self.w.value = w
        self.b.value = b
        self.slope.value = slope
        return torch.from_numpy(self.ccp(X))
    
    def g_dpp_form(self, x, h):
        pass

    def score_dpp_form(self, x, h):
        pass

    def score(self, x, w, b):
        return x@w + b

    def f_batch(self, x, w, b, slope):
        return 0.5*cp.norm(cp.vstack([np.ones(x.shape[0]), (slope*self.score(x, w, b) + 1)]), 2, axis=0)

    def g_batch(self, x, w, b, slope):
        return 0.5*cp.norm(cp.vstack([np.ones((1, x.shape[0])), cp.reshape((slope*self.score(x, w, b) - 1), (1, x.shape[0]))]), 2, axis=0)

    def c_batch(self, x, r, x_dim, scale):
        return (scale)*cp.square(cp.norm(x-r, 2, axis=1))

    def f_derivative_batch(self, x, w, b, slope):
        nablas = 0.5*slope*((slope*self.score(x, w, b) + 1)/cp.sqrt((slope*self.score(x, w, b) + 1)**2 + 1))
        return cp.reshape(nablas, (nablas.shape[0], 1))@cp.reshape(w, (1, x.shape[1]))
    
    def f(self, x, w, b, slope):
        return 0.5*cp.norm(cp.hstack([1, (slope*self.score(x, w, b) + 1)]), 2)

    def g(self, x, w, b, slope):
        return 0.5*cp.norm(cp.hstack([1, (slope*self.score(x, w, b) - 1)]), 2)

    def c(self, x, r, x_dim, scale):
        return (scale)*cp.sum_squares(x-r)

    def f_derivative(self, x, w, b, slope):
        return 0.5*cp.multiply(slope*((slope*self.score(x, w, b) + 1)/cp.sqrt((slope*self.score(x, w, b) + 1)**2 + 1)), w)
