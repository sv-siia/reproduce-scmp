from abc import ABC, abstractmethod
import cvxpy as cp
import numpy as np
from cvxpylayers.torch import CvxpyLayer
from strategic_classification.config.constants import X_LOWER_BOUND, X_UPPER_BOUND


class BaseDelta(ABC):
    """Abstract Base Class for solving convex-concave problems using CVXPY."""
    def __init__(self, x_dim):
        self.x_dim = x_dim
        self.x = cp.Variable(x_dim)
        self.r = cp.Parameter(x_dim, value=np.random.randn(x_dim))
        self.f_der = cp.Parameter(x_dim, value=np.random.randn(x_dim))

        self.h__w_hh_hy = cp.Parameter(1, value=np.random.randn(1))
        self.w_xh_hy = cp.Parameter(x_dim, value=np.random.randn(x_dim))
        self.w_b_hy = cp.Parameter(1, value=np.random.randn(1))

        target = self.x @ self.f_der - self.g_dpp_form(self.x) - self.c(self.x, self.r)
        constraints = [self.x >= X_LOWER_BOUND, self.x <= X_UPPER_BOUND]
        objective = cp.Maximize(target)
        problem = cp.Problem(objective, constraints)

        self.layer = CvxpyLayer(problem, parameters=[self.r, self.h__w_hh_hy, self.w_xh_hy, self.w_b_hy, self.f_der],
                                variables=[self.x])

    @abstractmethod
    def g_dpp_form(self, x):
        """Compute g in DPP form. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def c(self, x, r):
        """Constraint function. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def optimize_X(self, *args, **kwargs):
        """
        Optimization logic specific to each model.
        Must be implemented by subclasses.
        """
        pass
