import cvxpy as cp
import numpy as np
from cvxpylayers.torch import CvxpyLayer
from src.strategic_classification.config.constants import X_LOWER_BOUND, X_UPPER_BOUND, TRAIN_SLOPE


class DELTA():
    """Class for solving the convex-concave problem using CVXPY."""
    def __init__(self, x_dim, h_dim, funcs):
        self.g = funcs["g_dpp_form"]
        self.c = funcs["c"]
        
        self.x = cp.Variable(x_dim)
        self.r = cp.Parameter(x_dim, value = np.random.randn(x_dim))
        self.h__w_hh_hy = cp.Parameter(1, value = np.random.randn(1))
        self.w_xh_hy = cp.Parameter(x_dim, value = np.random.randn(x_dim))
        self.w_b_hy = cp.Parameter(1, value = np.random.randn(1))
        self.f_der = cp.Parameter(x_dim, value = np.random.randn(x_dim))

        target = self.x@self.f_der-self.g(self.x, self.h__w_hh_hy, self.w_xh_hy, self.w_b_hy, TRAIN_SLOPE)-self.c(self.x, self.r)
        constraints = [self.x >= X_LOWER_BOUND,
                       self.x <= X_UPPER_BOUND]
        objective = cp.Maximize(target)
        problem = cp.Problem(objective, constraints)
        self.layer = CvxpyLayer(problem, parameters=[self.r, self.h__w_hh_hy, self.w_xh_hy, self.w_b_hy, self.f_der],
                                variables=[self.x])
        
        
    def optimize_X(self, X, H, w_hy, w_hh, w_xh, b, F_DER):
        h__w_hh_hy = H@(w_hy@w_hh).T
        h__w_hh_hy = h__w_hh_hy.reshape(h__w_hh_hy.size()[0], 1)
        w_xh_hy = w_hy@w_xh
        w_b_hy = b@w_hy.T
        w_b_hy = w_b_hy.reshape(1)
        return self.layer(X, h__w_hh_hy, w_xh_hy, w_b_hy, F_DER)[0]
