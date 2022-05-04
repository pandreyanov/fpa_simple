import numpy as np
import numba as nb
from scipy.signal import fftconvolve

def q_smooth(sorted_bids, kernel, sample_size, band, i_band, trim, is_sorted = False, paste_ends = False, reflect = False):
    
    if is_sorted == False:
        sorted_bids = np.sort(sorted_bids)
    
    spacings = sorted_bids - np.roll(sorted_bids,1)
    spacings[0] = 0
    
    if reflect == False:
        mean = spacings.mean()
        out = (fftconvolve(spacings-mean, kernel, mode = 'same') + mean)*sample_size
    
    if reflect == True:
        reflected = np.concatenate((np.flip(spacings[:trim]), spacings, np.flip(spacings[-trim:])))
        out = fftconvolve(reflected, kernel, mode = 'same')[trim:-trim]*sample_size
    
    if paste_ends == True:
        out[:trim] = out[trim]
        out[-trim:] = out[-trim]
    
    return out

@nb.jit(nopython = True)
def binning(bids, sample_size):
    histogram = np.zeros(sample_size)
    for k in range(sample_size):
        histogram[int(sample_size*bids[k])] += 1
    return histogram

def f_smooth(bids, kernel, sample_size, band, i_band, trim, paste_ends = False, reflect = False):
    histogram = binning(bids, sample_size)
    
    if reflect == False:
        mean = histogram.mean()
        out = fftconvolve(histogram - mean, kernel, mode = 'same') + mean
        
    if reflect == True:
        reflected = np.concatenate((np.flip(histogram[:trim]), histogram, np.flip(histogram[-trim:])))
        out = fftconvolve(reflected, kernel, mode = 'same')[trim:-trim]
    
    if paste_ends == True:
        out[:trim] = out[trim]
        out[-trim:] = out[-trim]
    
    return out

def v_smooth(hat_Q, hat_q, A_4):
    return hat_Q + A_4*hat_q