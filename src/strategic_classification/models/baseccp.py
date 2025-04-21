from abc import ABC, abstractmethod
import cvxpy as cp

class BaseCCP(ABC):
    """Base class for solving the convex-concave problem using CVXPY."""
    def __init__(self, x_dim, h_dim):
        self.x = cp.Variable(x_dim)
        self.xt = cp.Parameter(x_dim)
        self.r = cp.Parameter(x_dim)
        self.h = cp.Parameter(h_dim)
        self.w_hy = cp.Parameter(h_dim)
        self.w_hh = cp.Parameter((h_dim, h_dim))
        self.w_xh = cp.Parameter((h_dim, x_dim))
        self.slope = cp.Parameter(1)
        self.b = cp.Parameter(h_dim)

    @abstractmethod
    def f_derivative(self, *args, **kwargs):
        """Compute the derivative of the function f."""
        pass

    @abstractmethod
    def c(self, x):
        """Constraint function."""
        pass

    @abstractmethod
    def g_dpp_form(self, x, h):
        """Compute g in DPP form."""
        pass

    @abstractmethod
    def g(self, x, h):
        """Compute g."""
        pass

    @abstractmethod
    def score_dpp_form(self, x, h):
        """Compute score in DPP form."""
        pass

    @abstractmethod
    def score(self, x, h):
        """Compute score."""
        pass

    @abstractmethod
    def optimize_X(self, *args, **kwargs):
        """
        Optimization logic common to all models.
        Each custom model should implement its own version of this method.
        """
        pass
