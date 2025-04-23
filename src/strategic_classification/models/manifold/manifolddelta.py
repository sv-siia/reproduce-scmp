import cvxpy as cp
import numpy as np
from cvxpylayers.torch import CvxpyLayer

TRAIN_SLOPE = 1
X_LOWER_BOUND = -30
X_UPPER_BOUND = 30

class DELTA:

    def __init__(self, x_dim, h_dim, funcs):
        self.g = funcs["g"]
        self.c = funcs["c"]

        self.x = cp.Variable(x_dim)
        self.r = cp.Parameter(x_dim, value = np.random.randn(x_dim))
        self.w = cp.Parameter(x_dim, value = np.random.randn(x_dim))
        self.b = cp.Parameter(1, value = np.random.randn(1))
        self.f_der = cp.Parameter(x_dim, value = np.random.randn(x_dim))

        target = self.x@self.f_der-self.g(self.x, self.w, self.b, TRAIN_SLOPE)-self.c(self.x, self.r, x_dim)
        constraints = [self.x >= X_LOWER_BOUND,
                       self.x <= X_UPPER_BOUND]
        objective = cp.Maximize(target)
        problem = cp.Problem(objective, constraints)
        self.layer = CvxpyLayer(problem, parameters=[self.r, self.w, self.b, self.f_der],
                                variables=[self.x])
    
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
    def optimize_X(self, X, w, b, F_DER, B_SPAN):
        return self.layer(X, w, b, F_DER)[0]

class DELTA_MANIFOLD(DELTA):

    def __init__(self, x_dim, h_dim, funcs):
        self.g = funcs["g"]
        self.c = funcs["c"]

        self.x = cp.Variable(x_dim)
        self.v = cp.Variable(h_dim)
        self.r = cp.Parameter(x_dim, value = np.random.randn(x_dim))
        self.w = cp.Parameter(x_dim, value = np.random.randn(x_dim))
        self.b = cp.Parameter(1, value = np.random.randn(1))
        self.f_der = cp.Parameter(x_dim, value = np.random.randn(x_dim))
        self.B_span = cp.Parameter((x_dim, h_dim), value = np.random.randn(x_dim, h_dim))

        target = self.x@self.f_der-self.g(self.x, self.w, self.b, TRAIN_SLOPE)-self.c(self.x, self.r, x_dim)
        constraints = [self.x >= X_LOWER_BOUND,
                       self.x <= X_UPPER_BOUND,
                      self.B_span@self.v == self.x-self.r]
        objective = cp.Maximize(target)
        problem = cp.Problem(objective, constraints)
        self.layer = CvxpyLayer(problem, parameters=[self.r, self.w, self.b, self.f_der, self.B_span],
                                variables=[self.x, self.v])

    def optimize_X(self, X, w, b, F_DER, B_SPAN):
        return self.layer(X, w, b, F_DER, B_SPAN)[0]
