import cvxpy as cp
import numpy as np
from cvxpylayers.torch import CvxpyLayer
from strategic_classification.models.basedelta import BaseDelta

TRAIN_SLOPE = 1
EVAL_SLOPE = 5
X_LOWER_BOUND = -10
X_UPPER_BOUND = 10

class VanilaDelta(BaseDelta):
    """Class for solving the convex-concave problem using CVXPY."""
    def __init__(self, x_dim, scale):
        
        self.x = cp.Variable(x_dim)
        self.r = cp.Parameter(x_dim, value = np.random.randn(x_dim))
        self.w = cp.Parameter(x_dim, value = np.random.randn(x_dim))
        self.b = cp.Parameter(1, value = np.random.randn(1))
        self.f_der = cp.Parameter(x_dim, value = np.random.randn(x_dim))

        target = self.x@self.f_der-self.g(self.x, self.w, self.b, TRAIN_SLOPE)-self.c_Quad(self.x, self.r, x_dim, scale)
        constraints = [self.x >= X_LOWER_BOUND,
                       self.x <= X_UPPER_BOUND]
        objective = cp.Maximize(target)
        problem = cp.Problem(objective, constraints)
        self.layer = CvxpyLayer(problem, parameters=[self.r, self.w, self.b, self.f_der],
                                variables=[self.x])
        
    def optimize_X(self, X, w, b, F_DER):
        return self.layer(X, w, b, F_DER)[0]
    
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
    
    def g_dpp_form(self, x):
        """Compute g in DPP form. Must be implemented by subclasses."""
        pass

    def c(self, x, r):
        """Constraint function. Must be implemented by subclasses."""
        pass
