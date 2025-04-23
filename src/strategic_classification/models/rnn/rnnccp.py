import cvxpy as cp
import torch
import numpy as np
from strategic_classification.config.constants import X_LOWER_BOUND, X_UPPER_BOUND
from strategic_classification.models.baseccp import BaseCCP


class RNNCCP(BaseCCP):
    """Class for solving the convex-concave problem using CVXPY."""
    def __init__(self, x_dim, h_dim):
        super().__init__(x_dim)
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
    
    def score(self, x, h, w_hy, w_hh, w_xh, b):
        return (h@w_hh.T + x@w_xh.T + b)@w_hy.T
    
    def score_dpp_form(self, x, h__w_hh_hy, w_xh_hy, w_b_hy):
        return h__w_hh_hy + x@w_xh_hy.T + w_b_hy

    def g(self, x, h, w_hy, w_hh, w_xh, b, slope):
        return 0.5*cp.norm(cp.hstack([1, (slope*self.score(x, h, w_hy, w_hh, w_xh, b) - 1)]), 2)

    def g_dpp_form(self, x, h__w_hh_hy, w_xh_hy, w_b_hy, slope):
        return 0.5*cp.norm(cp.hstack([1, (slope*self.score_dpp_form(x, h__w_hh_hy, w_xh_hy, w_b_hy) - 1)]), 2)

    def c(self, x, r):
        return cp.sum_squares(x-r)

    def f_derivative(self, x, h, w_hy, w_hh, w_xh, b, slope):
        return 0.5*slope*((slope*self.score(x, h, w_hy, w_hh, w_xh, b) + 1)
                            /cp.sqrt((slope*self.score(x, h, w_hy, w_hh, w_xh, b) + 1)**2 + 1))*(w_hy@w_xh)
        
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
