
import numpy as np

def build_poly(x, degree):
    powers = np.arange(0,degree+1)[:, None]
    mat = x**powers
    return mat.transpose()
    
  

  