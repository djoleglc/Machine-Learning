import numpy as np

def compute_loss(y, tx, w):
        fitted = tx @ w
        n = y.shape[0]
        return 1/(2*n) * ((y-fitted)**2).sum()
