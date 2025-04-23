from abc import ABC, abstractmethod
import cvxpy as cp

class BaseCCP(ABC):
    """Base class for solving the convex-concave problem using CVXPY."""
    def __init__(self, x_dim):
        self.x = cp.Variable(x_dim)
        self.xt = cp.Parameter(x_dim)
        self.r = cp.Parameter(x_dim)

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
