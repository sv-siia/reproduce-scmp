import cvxpy as cp
import numpy as np
from cvxpylayers.torch import CvxpyLayer
from strategic_classification.models.basedelta import BaseDelta
from strategic_classification.config.constants import X_LOWER_BOUND, X_UPPER_BOUND


XDIM = 11
COST = 1/XDIM
TRAIN_SLOPE = 1

class DeltaRecourse(BaseDelta):
    
    def __init__(self, x_dim):
        self.x = cp.Variable(x_dim)
        self.r = cp.Parameter(x_dim, value = np.random.randn(x_dim))
        self.w = cp.Parameter(x_dim, value = np.random.randn(x_dim))
        self.b = cp.Parameter(1, value = np.random.randn(1))
        self.f_der = cp.Parameter(x_dim, value = np.random.randn(x_dim))

        target = self.x@self.f_der-self.g(self.x, self.w, self.b, TRAIN_SLOPE)-self.c(self.x, self.r)
        constraints = [self.x >= X_LOWER_BOUND,
                       self.x <= X_UPPER_BOUND]
        objective = cp.Maximize(target)
        problem = cp.Problem(objective, constraints)
        self.layer = CvxpyLayer(problem, parameters=[self.r, self.w, self.b, self.f_der],
                                variables=[self.x])
        
        
    def optimize_X(self, X, w, b, F_DER):
        return self.layer(X, w, b, F_DER)[0]
    
    def c(self, x, r, COST=COST):
        return COST * cp.sum_squares(x - r)

    def g(self, x, w, b, slope):
        return 0.5*cp.norm(cp.hstack([1, (slope* self.score(x, w, b) - 1)]), 2)

