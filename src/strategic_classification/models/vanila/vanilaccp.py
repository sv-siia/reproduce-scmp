import cvxpy as cp
import torch
import numpy as np
from strategic_classification.models.baseccp import BaseCCP

X_LOWER_BOUND = -10
X_UPPER_BOUND = 10

class VanilaCCP(BaseCCP):
    def __init__(self, x_dim, scale):
        super().__init__(x_dim)
        self.x = cp.Variable(x_dim)
        self.xt = cp.Parameter(x_dim)
        self.r = cp.Parameter(x_dim)
        self.w = cp.Parameter(x_dim)
        self.b = cp.Parameter(1)
        self.slope = cp.Parameter(1)

        target = self.x@self.f_derivative(self.xt, self.w, self.b, self.slope)-self.g(self.x, self.w, self.b, self.slope)-self.c_Quad(self.x, self.r, x_dim, scale)
        constraints = [self.x >= X_LOWER_BOUND,
                       self.x <= X_UPPER_BOUND]
        self.prob = cp.Problem(cp.Maximize(target), constraints)
        
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
            diff = np.linalg.norm(self.x.value - self.xt.value)
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
        
        return torch.stack([torch.from_numpy(self.ccp(x)) for x in X])
    
    def score(self, x, w, b):
        return x@w + b

    def f(self, x, w, b, slope):
        return 0.5*cp.norm(cp.hstack([1, (slope*self.score(x, w, b) + 1)]), 2)

    def g(self, x, w, b, slope):
        return 0.5*cp.norm(cp.hstack([1, (slope*self.score(x, w, b) - 1)]), 2)

    def c_Quad(self, x, r, x_dim, scale):
        return (scale)*cp.sum_squares(x-r)

    def f_derivative(self, x, w, b, slope):
        return 0.5*cp.multiply(slope*((slope*self.score(x, w, b) + 1)/cp.sqrt((slope*self.score(x, w, b) + 1)**2 + 1)), w)

    def g_dpp_form(self, x, h):
        """Compute g in DPP form."""
        pass

    def score_dpp_form(self, x, h):
        """Compute score in DPP form."""
        pass

    def c(self, x):
        """Constraint function."""
        pass
    
