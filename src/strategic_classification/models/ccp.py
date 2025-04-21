import cvxpy as cp
import torch
import numpy as np
from src.strategic_classification.config.constants import X_LOWER_BOUND, X_UPPER_BOUND


class CCP:
    """Class for solving the convex-concave problem using CVXPY."""
    def __init__(self, x_dim, h_dim, funcs):
        self.f_derivative = funcs["f_derivative"]
        self.g = funcs["g"]
        self.c = funcs["c"]
        
        self.x = cp.Variable(x_dim)
        self.xt = cp.Parameter(x_dim)
        self.r = cp.Parameter(x_dim)
        self.h = cp.Parameter(h_dim)
        self.w_hy = cp.Parameter(h_dim)
        self.w_hh = cp.Parameter((h_dim, h_dim))
        self.w_xh = cp.Parameter((h_dim, x_dim))
        self.slope = cp.Parameter(1)
        self.b = cp.Parameter(h_dim)

        target = self.x@self.f_derivative(self.xt, self.h, self.w_hy, self.w_hh, self.w_xh, self.b, self.slope)-self.g(self.x, self.h, self.w_hy, self.w_hh, self.w_xh, self.b, self.slope)-self.c(self.x, self.r)
        constraints = [self.x >= X_LOWER_BOUND,
                       self.x <= X_UPPER_BOUND]
        self.prob = cp.Problem(cp.Maximize(target), constraints)
        
    def ccp(self, r, h):
        """
        numpy to numpy
        """
        self.xt.value = r
        self.r.value = r
        self.h.value = h
        result = self.prob.solve()
        diff = np.linalg.norm(self.xt.value - self.x.value)
        cnt = 0
        while diff > 0.0001 and cnt < 100:
            cnt += 1
            self.xt.value = self.x.value
            result = self.prob.solve()
            diff = np.linalg.norm(self.x.value - self.xt.value)
        return self.x.value
    
    def optimize_X(self, X, H, w_hy, w_hh, w_xh, b, slope):
        """
        tensor to tensor
        """
        w_hy = w_hy.detach().numpy()
        w_hh = w_hh.detach().numpy()
        w_xh = w_xh.detach().numpy()
        b = b.detach().numpy()
        slope = np.full(1, slope)
        X = X.numpy()
        H = H.detach().numpy()
        
        self.w_hy.value = w_hy
        self.w_hh.value = w_hh
        self.w_xh.value = w_xh
        self.b.value = b
        self.slope.value = slope
        
        return torch.stack([torch.from_numpy(self.ccp(x, h)) for x, h in zip(X, H)])
