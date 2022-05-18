import numpy as np
from scipy.stats import norm
import os
from multiprocess import Pool

def make_ci(self, confidence):
    
    one = norm.ppf(confidence/100)
    two = norm.ppf((confidence+(100-confidence)/2)/100)
    
    self.ci_one = np.sqrt(self.intKsq)*one
    self.ci_two = np.sqrt(self.intKsq)*two
    
def make_cb(self, confidence, draws):

    def simulate_q(i): 
        np.random.seed(i)
        mc = np.sort(np.random.uniform(0, 1, self.sample_size))
        return q_smooth(mc, self.kernel, *self.band_options, reflect = True, is_sorted = True)

#     p = Pool(os.cpu_count())
#     hat_qs = np.array(p.map(simulate_q, range(draws)))
#     p.close()
#     p.join()
    
#     sup = np.max((hat_qs-1)[:,self.trim:-self.trim], axis = 1)
#     supabs = np.max(np.abs(hat_qs-1)[:,self.trim:-self.trim], axis = 1)

#     self.cb_one = np.percentile(sup, confidence)
#     self.cb_two = np.percentile(supabs, confidence+(100-confidence)/2)