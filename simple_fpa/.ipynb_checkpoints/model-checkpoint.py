import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sb

from .kernels import *
from .estimators import *

class Model:
    '''A package for the "Nonparametric inference on counterfactuals in sealed first-price auctions" paper.'''
    def __init__(self, data, auction_ids, bids):
        self.data = data
        self.auction_ids = auction_ids
        self.bids = bids
        
    def count_bidders_by_auction_id(self):
        self.data.sort_values(by = self.auction_ids, inplace = True)
        
        self.data['__ones'] = 1
        self.data['_bidders'] = self.data.groupby(by = self.auction_ids)['__ones'].transform(sum)
        self.data.drop(columns = ['__ones'], inplace = True)
        
        frec = self.data.groupby(by = 'auctionid')._bidders.first().value_counts().values
        frec = frec/np.sum(frec)
        n_bids = self.data.groupby(by = 'auctionid')._bidders.first().value_counts().index.values
        self.frec = {int(i):j for i,j in zip(n_bids, frec)}
        
    def fit(self, cont_covs, disc_covs, model_type = 'multiplicative'):
        self.cont_covs = cont_covs
        self.disc_covs = disc_covs
        self.model_type = model_type
        
        if self.model_type == 'multiplicative':
            self.formula = 'np.log(' + self.bids + ') ~ '
            for c in self.cont_covs:
                self.formula += 'np.log(' + c + ') + '
            for c in self.disc_covs:
                self.formula += 'C(' + c + ') + '
            self.formula = self.formula[:-2]
            
        if self.model_type == 'additive':
            self.formula = self.bids + ' ~ '
            for c in self.cont_covs:
                self.formula += c + ' + '
            for c in self.disc_covs:
                self.formula += 'C(' + c + ') + '
            self.formula = self.formula[:-2]
            
        self.ols = smf.ols(formula=self.formula, data=self.data).fit()
        
        self.data = self.data.copy()
        
        if self.model_type == 'multiplicative':
            self.data['_resid'] = np.exp(self.ols.resid)
            self.data['_fitted'] = np.exp(self.ols.fittedvalues)
            
        if self.model_type == 'additive':
            self.data['_resid'] = self.ols.resid
            self.data['_fitted'] = self.ols.fittedvalues
            
    def summary(self, show_dummies = False):
        for row in self.ols.summary().as_text().split('\n'):
            if row[:2] != 'C(' or show_dummies == True:
                print(row)
                
    def trim_residuals(self, trim_percent = 5):
        left = np.percentile(self.data._resid.values, trim_percent)
        right = np.percentile(self.data._resid.values, 100-trim_percent)
        self.data = self.data[(self.data._resid > left) & (self.data._resid < right)]
        
    def calibrate_band(self, sample, u_trim, smoothing):
        sample_size = len(sample)
        std = np.std(sample)
        band = 1.06*std*np.power(sample_size, smoothing)
        delta = 1 # this is only if on [0,1]
        u_band = band/delta
        i_band = int(u_band*sample_size)
        trim = int(u_trim*sample_size)

        if trim < i_band:
            print('Warning: Not enough trimming, look out for boundary effects.')
    
        return sample_size, band, i_band, trim
    
    def calibrate_part(self, frec):
        
        M = np.max(list(self.frec.keys()))
        m_min = np.min(list(self.frec.keys()))

        A_1 = 0*self.u_grid
        A_1_prime = 0*self.u_grid
        a = 0
        
        for m, pm in self.frec.items():
            A_1 += pm*np.power(self.u_grid, m-1)
            A_1_prime += pm*(m-1)*np.power(self.u_grid, m-2)
            a += m*pm/M
            
        A_1_prime[1] = A_1_prime[0] # avoid division by zero

        A_2 = self.u_grid*A_1
        A_3 = (1-self.u_grid)*A_1
        
        A_4 = A_1/A_1_prime

        return M, A_1, A_2, A_3, A_4, a
    
    def q_smooth(self, sorted_bids, kernel, sample_size, band, i_band, trim, is_sorted = False, paste_ends = False, reflect = False):
        
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
    
    def calibrate(self, smoothing_rate = 0.2, trim_percent = 5, reflect = True):
        
        self.data = self.data.sort_values(by = '_resid').copy() # here comes the sorting
        
        self.obs = self.data._resid.values.copy()
        self.intercept = self.obs.min()
        self.obs -= self.intercept
        self.scale = self.obs.max()
        self.obs /= self.obs.max()
        
        self.smoothing = -smoothing_rate
        self.u_trim = trim_percent/100
        self.reflect = reflect
        
        self.band_options = self.calibrate_band(self.obs, self.u_trim, self.smoothing)
        self.sample_size, self.band, self.i_band, self.trim = self.band_options
        
        self.u_grid = np.linspace(0, 1, self.sample_size)
        self.kernel = make_kernel(self.i_band, kernel = tri)
        
        self.part_options = self.calibrate_part(self.frec)
        self.M, self.A_1, self.A_2, self.A_3, self.A_4, self.a = self.part_options
        
        self.hat_Q = self.intercept + self.scale*self.obs # they are sorted with the dataset
        
        self.hat_f = f_smooth(self.obs, self.kernel, *self.band_options, paste_ends = False, reflect = reflect)
        self.hat_q = self.scale*q_smooth(self.obs, self.kernel, *self.band_options, is_sorted = True, reflect = reflect)
        
        self.hat_v = v_smooth(self.hat_Q, self.hat_q, self.A_4)
        
    def plot_stats(self):
        fig, ((ax1, ax2),(ax3, ax4),(ax5, ax6)) = plt.subplots(3,2)
        sb.histplot(data = self.data._resid, stat = 'density', bins = 50, facecolor=(0, 0, 0, 0),
                       linewidth=1,
                       edgecolor='black', ax = ax2);
        sb.countplot(x = self.data.groupby(by = 'auctionid')._bidders.first().astype(int), facecolor=(0, 0, 0, 0),
                       linewidth=1,
                       edgecolor='black', ax = ax1)
        ax1.set_xlabel('bidders')
        ax2.set_xlabel('bid residuals')
        ax2.set_ylabel('density')
        
        ax3.plot(self.u_grid, self.A_1, label = '$A_1$')
        ax3.plot(self.u_grid, self.A_2, label = '$A_2$')
        ax3.plot(self.u_grid, self.A_3, label = '$A_3$')
        ax3.plot(self.u_grid, self.A_4, label = '$A_4$')
        ax3.legend()
        
        ax4.plot(self.u_grid, self.hat_q, label = 'smooth $\hat q(u)$')
        ax4.plot(self.u_grid, self.hat_f, label = 'smooth $\hat f(b)$')
        ax4.legend()
        
        avg_fitted = self.data._fitted.mean()
        
        if self.model_type == 'multiplicative':
            b_qf = self.hat_Q * avg_fitted
            v_qf = self.hat_v * avg_fitted
        
        if self.model_type == 'additive':
            b_qf = self.hat_Q + avg_fitted
            v_qf = self.hat_v + avg_fitted
        
        ax5.plot(self.u_grid, b_qf, label = 'avg bid q.f.')
        ax5.plot(self.u_grid, v_qf, label = 'avg value q.f.')
        ax5.legend()
        
        plt.tight_layout()
        plt.show()
        
    def predict(self):
        self.data['_latent_resid'] = self.hat_v 
        
        if self.model_type == 'multiplicative':
            self.data['_latent_'+self.bids] = self.data['_latent_resid']*self.data._fitted
            
        if self.model_type == 'additive':
            self.data['_latent_'+self.bids] = self.data['_latent_resid']+self.data._fitted
            
    def fit_predict(self, model_type = 'multiplicative'):
        self.fit(self.cont_covs, self.disc_covs, model_type)
        self.summary()
        self.trim_residuals(5)
        self.calibrate()
        self.predict()
        self.plot_stats()
        
        
        
        
    
        
        