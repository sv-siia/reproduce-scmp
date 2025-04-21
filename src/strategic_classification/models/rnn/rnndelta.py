import cvxpy as cp
from strategic_classification.models.basedelta import BaseDelta

class DeltaRNN(BaseDelta):
    def g_dpp_form(self, x):
        """Override g in DPP form for RNN."""
        return 0.5 * cp.norm(cp.hstack([1, (self.h__w_hh_hy + x @ self.w_xh_hy.T + self.w_b_hy - 1)]), 2)

    def c(self, x, r):
        """Override constraint function for RNN."""
        return cp.sum_squares(x - r)

    def optimize_X(self, X, H, w_hy, w_hh, w_xh, b, F_DER):
        """RNN-specific optimization logic."""
        h__w_hh_hy = H @ (w_hy @ w_hh).T
        h__w_hh_hy = h__w_hh_hy.reshape(h__w_hh_hy.size()[0], 1)
        w_xh_hy = w_hy @ w_xh
        w_b_hy = b @ w_hy.T
        w_b_hy = w_b_hy.reshape(1)
        return self.layer(X, h__w_hh_hy, w_xh_hy, w_b_hy, F_DER)[0]
