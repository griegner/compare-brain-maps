import numpy as np


def quantile_test(param_, params_, alpha=0.05):
    """test whether `param_` is inside the (alpha/2, 1-alpha/2) quantile interval."""
    lwr, upr = np.quantile(params_, [alpha / 2, 1 - alpha / 2])
    return bool(lwr <= param_ <= upr)
