import cvxpy as cp

# This file defines gain and cost functions using CVXPY for optimization tasks.
# It includes functions for scoring, gain calculation, cost calculation, and their derivatives.

def score(x, w, b):
    return x@w + b

def f(x, w, b, slope):
    return 0.5*cp.norm(cp.hstack([1, (slope*score(x, w, b) + 1)]), 2)

def g(x, w, b, slope):
    return 0.5*cp.norm(cp.hstack([1, (slope*score(x, w, b) - 1)]), 2)

def f_derivative(x, w, b, slope):
    return 0.5*cp.multiply(slope*((slope*score(x, w, b) + 1)/cp.sqrt((slope*score(x, w, b) + 1)**2 + 1)), w)

def score_dpp_form(x, h__w_hh_hy, w_xh_hy, w_b_hy):
    return h__w_hh_hy + x@w_xh_hy.T + w_b_hy

def g_dpp_form(x, h__w_hh_hy, w_xh_hy, w_b_hy, slope):
    return 0.5*cp.norm(cp.hstack([1, (slope*score_dpp_form(x, h__w_hh_hy, w_xh_hy, w_b_hy) - 1)]), 2)

def c(x, r):
    return cp.sum_squares(x-r)
