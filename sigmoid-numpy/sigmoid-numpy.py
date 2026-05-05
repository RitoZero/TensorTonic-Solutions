import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    x = np.asarray(x, dtype=float)

    result = np.empty_like(x)

    pos_mask = x >= 0
    neg_mask = ~pos_mask

    result[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
    result[neg_mask] = np.exp(x[neg_mask]) / (1 + np.exp(x[neg_mask]))

    return result
    