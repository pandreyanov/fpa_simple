import numpy as np
import numba as nb
import scipy as sp

@nb.jit(nopython=True)
def tri(u):
    return np.maximum(35*np.power(1 - np.power(u, 2), 3)/32, 0)

@nb.jit(nopython=True)
def epa(u):
    return np.maximum(3*(1 - np.power(u, 2))/4, 0)

@nb.jit(nopython=True)
def rec(u):
    return (np.sign(1/2 - np.abs(u)) + 1)/2

@nb.jit(nopython=True)
def make_kernel(i_band, kernel = tri):
    return np.array([kernel(j/i_band)/i_band for j in range(-i_band+1, i_band)])