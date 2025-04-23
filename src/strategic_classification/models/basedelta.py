from abc import ABC, abstractmethod


class BaseDelta(ABC):
    """Abstract Base Class for solving convex-concave problems using CVXPY."""
    
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
