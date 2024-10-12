import numpy as np
from scipy.signal import fftconvolve

def q_smooth(sorted_bids, kernel, sample_size, band, i_band, trim, is_sorted = False, boundary = 'reflect'):
    
    if is_sorted == False:
        sorted_bids = np.sort(sorted_bids)
    
    spacings = sorted_bids - np.roll(sorted_bids,1)
    spacings[0] = 0
    
    if boundary == 'mean':
        mean = spacings.mean()
        out = (fftconvolve(spacings-mean, kernel, mode = 'same') + mean)*sample_size
    
    if boundary == 'reflect':
        reflected = np.concatenate((np.flip(spacings[:trim]), spacings, np.flip(spacings[-trim:])))
        out = fftconvolve(reflected, kernel, mode = 'same')[trim:-trim]*sample_size
    
    if boundary == 'constant':
        out = fftconvolve(spacings, kernel, mode = 'same')*sample_size
        out[:trim] = out[trim]
        out[-trim:] = out[-trim]
        
    if boundary == 'zero':
        out = fftconvolve(spacings, kernel, mode = 'same')*sample_size
        out[:trim] = 0
        out[:trim] = 0
        out[-trim:] = 0
        out[-trim:] = 0
    
    return out

def f_smooth(bids, kernel, sample_size, band, i_band, trim, paste_ends = False, boundary = 'reflect'):
    histogram, _ = np.histogram(bids, sample_size, range = (0,1))
    
    if boundary == 'mean':
        mean = histogram.mean()
        out = (fftconvolve(histogram-mean, kernel, mode = 'same') + mean)
    
    if boundary == 'reflect':
        reflected = np.concatenate((np.flip(histogram[:trim]), histogram, np.flip(histogram[-trim:])))
        out = fftconvolve(reflected, kernel, mode = 'same')[trim:-trim]
    
    if boundary == 'constant':
        out = fftconvolve(histogram, kernel, mode = 'same')
        out[:trim] = out[trim]
        out[-trim:] = out[-trim]
        
    if boundary == 'zero':
        out = fftconvolve(histogram, kernel, mode = 'same')
        out[:trim] = 0
        out[:trim] = 0
        out[-trim:] = 0
        out[-trim:] = 0
    
    return out

def v_smooth(hat_Q, hat_q, A_4):
    return hat_Q + A_4*hat_q

def d(arr):
    diff = arr - np.roll(arr, 1)
    diff[0] = diff[1]
    return diff*len(diff)
    
def int_lowbound(arr):
    return np.flip(np.cumsum(np.flip(arr)))/len(arr)

def int_uppbound(arr):
    return np.cumsum(arr)/len(arr)

def total_surplus(v, Mtilde, A_1, A_2, A_3, A_4):
    return int_lowbound(v*d(A_2))

def bidder_surplus(v, Mtilde, A_1, A_2, A_3, A_4):
    return int_lowbound(A_3*d(v))

def revenue(v, Mtilde, A_1, A_2, A_3, A_4):
    return total_surplus(v, Mtilde, A_1, A_2, A_3, A_4) - Mtilde*bidder_surplus(v, Mtilde, A_1, A_2, A_3, A_4)

def total_surplus_from_Q(Q, trim, Mtilde, A_1, A_2, A_3, A_4):
    psi = d(A_2)
    chi = psi - d(A_4*psi)
    return A_4[-trim-1]*psi[-trim-1]*Q[-trim-1] - A_4*psi*Q + int_lowbound(chi*Q)

def revenue_from_Q_and_v(Q, v, trim, Mtilde, A_1, A_2, A_3, A_4):
    phi = Mtilde*A_3
    psi = d(A_2 + Mtilde*A_3)
    chi = psi - d(A_4*psi)
    return phi*v + A_4[-trim-1]*psi[-trim-1]*Q[-trim-1] - A_4*psi*Q + int_lowbound(chi*Q)

# need to fix this aftert he introduction of Mtilde (Mtilde = a M)
# def bidder_surplus_from_Q_and_v(Q, v, trim, M, A_1, A_2, A_3, A_4, a):
#     phi = -a*A_3
#     psi = -d(a*A_3)
#     chi = psi - d(A_4*psi)
#     return phi*v + A_4[-trim-1]*psi[-trim-1]*Q[-trim-1] - A_4*psi*Q + int_lowbound(chi*Q)