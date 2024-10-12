import numpy as np
from scipy.stats import norm
import os
from multiprocess import Pool
#from pathos.multiprocessing import ProcessingPool as Pool

from .estimators import *

def add_column(self, name, values):
    self.data[name] = np.nan
    self.data.loc[self.active_index, name] = values
    
def make_ci_asy(self, confidence, hyp):
    
    one = norm.ppf(confidence/100)
    two = norm.ppf((confidence+(100-confidence)/2)/100)
    
    self.ci_one = np.sqrt(self.intKsq)*one
    self.ci_two = np.sqrt(self.intKsq)*two
    
    if hyp == 'twosided':
        self.ci = self.ci_two
    if hyp == 'onesided':
        self.ci = self.ci_one

    self.core_ci = self.ci*self.hat_q/np.sqrt(self.sample_size*self.band)
        
    add_column(self, '_q_ci_asy', self.core_ci)
    add_column(self, '_v_ci_asy', self.A_4*self.core_ci)
    add_column(self, '_bs_ci_asy', self.A_3*self.A_4*self.core_ci)
    add_column(self, '_rev_ci_asy', self.Mtilde*self.A_3*self.A_4*self.core_ci)
    
def make_cicb(self, confidence, draws, hyp, boundary):

    def simulate_Q(i): 
        np.random.seed(i)
        mc = np.sort(np.random.uniform(0, 1, self.sample_size))
        return self.hat_q*(mc-self.u_grid) # this is not uniform
    
    p = Pool(os.cpu_count())
    delta_Qs = np.array(p.map(simulate_Q, range(draws)))
    p.close()
    p.join()
    
    def simulate_q(i): 
        np.random.seed(i)
        mc = np.sort(np.random.uniform(0, 1, self.sample_size))
        mcq = q_smooth(mc, self.kernel, *self.band_options, is_sorted = True, boundary = boundary)
        return (mcq-1) # this is uniform

    p = Pool(os.cpu_count())
    delta_qs = np.array(p.map(simulate_q, range(draws)))
    p.close()
    p.join()
    
    if boundary == 'zero':
        delta_qs[:,-self.trim:] = 0
        delta_qs[:,:self.trim] = 0
        delta_Qs[:,-self.trim:] = 0
        delta_Qs[:,:self.trim] = 0
        
    if hyp == 'twosided':
        def _sup(x):
            return np.max(np.abs(x)[:,self.trim:-self.trim], axis = 1)
        def _perc(x):
            return np.percentile(x, confidence+(100-confidence)/2, axis = 0) # shall I trim it?
    if hyp == 'onesided':
        def _sup(x):
            return np.max(np.abs(x)[:,self.trim:-self.trim], axis = 1)
        def _perc(x):
            return np.percentile(x, confidence, axis = 0) # shall I trim it?
        
    delta_ts = np.apply_along_axis(lambda x: total_surplus_from_Q(x, self.trim, *self.part_options), 1, delta_Qs)
    del(delta_Qs)
       
    add_column(self, '_ts_ci', _perc(delta_ts))
    add_column(self, '_ts_cb', _perc(_sup(delta_ts)))
        
    core_ci = self.hat_q*_perc(delta_qs)
    core_cb = self.hat_q*_perc(_sup(delta_qs))
    del(delta_qs)
    
    add_column(self, '_q_ci', core_ci)
    add_column(self, '_q_cb', core_cb)
    
    add_column(self, '_v_ci', self.A_4*core_ci)
    add_column(self, '_v_cb', self.A_4*core_cb)

    self.bs_ci = self.A_3*self.A_4*core_ci
    self.bs_cb = self.A_3*self.A_4*core_cb
    
    add_column(self, '_bs_ci', self.bs_ci)
    add_column(self, '_bs_cb', self.bs_cb)

    self.rev_ci = self.Mtilde*self.A_3*self.A_4*core_ci
    self.rev_cb = self.Mtilde*self.A_3*self.A_4*core_cb
    
    add_column(self, '_rev_ci', self.rev_ci)
    add_column(self, '_rev_cb', self.rev_cb)
