import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sb
from pylab import rcParams

def plot_counterfactuals(self):
    rcParams['figure.figsize'] = 7, 7/3
    fig, (ax1, ax2) = plt.subplots(1,2)
    
    ax1.plot(self.u_grid, self.ts, label = 'total surplus')
    ax1.plot(self.u_grid, self.M*self.bs, label = 'bidders surplus (times M)')
    ax1.plot(self.u_grid, self.rev, label = 'revenue')
    
    ax1.legend()
    ax1.set_ylabel('in terms of residuals')
    
    plt.tight_layout()
    plt.show()

def plot_stats(self):
    rcParams['figure.figsize'] = 7, 7
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2)
    sb.countplot(x = self.data.groupby(by = 'auctionid')._bidders.first().astype(int), 
                 facecolor=(0, 0, 0, 0),
                 linewidth=1,
                 edgecolor='black', 
                 ax = ax1)
    
    ax1.set_xlabel('bidders')
    
    sb.histplot(data = self.data.loc[self.active_index,'_resid'], 
                stat = 'density', 
                bins = 50, 
                facecolor=(0, 0, 0, 0),
                linewidth=1,
                edgecolor='black', 
                ax = ax2);
    
    ax2.set_xlabel('bid residuals')
    ax2.set_ylabel('density')
    
    ax2.plot(self.intercept+self.scale*self.u_grid, 
             self.hat_f, 
             color = 'red', 
             label = 'smooth $\hat f(b)$', 
             linewidth=1)
    
    ax2.legend()
    
    ax3.plot(self.u_grid, self.A_1, label = '$A_1$')
    ax3.plot(self.u_grid, self.A_2, label = '$A_2$')
    ax3.plot(self.u_grid, self.A_3, label = '$A_3$')
    ax3.plot(self.u_grid, self.A_4, label = '$A_4$')
    ax3.legend()
        
    ax4.plot(self.u_grid, 
             self.hat_q, 
             label = 'smooth $\hat q(u)$', linewidth = 1, color = 'blue')
    ax4.plot(self.u_grid, self.hat_q+self.ci_two*self.hat_q, linestyle = '--', linewidth = 1, color = 'blue')
    ax4.plot(self.u_grid, self.hat_q-self.ci_two*self.hat_q, linestyle = '--', linewidth = 1, color = 'blue')
    
    
    ax4.plot(self.u_grid, 
             self.hat_f*self.scale, 
             color = 'red', 
             label = 'smooth $\hat f(b)$ (scale matched)', 
             linewidth=1)
    
    ax4.plot(self.u_grid, 
             (self.hat_f+self.ci_two*np.sqrt(self.hat_f))*self.scale, 
             color = 'red', 
             linewidth=1, linestyle = '--')
    
    ax4.plot(self.u_grid, 
             (self.hat_f-self.ci_two*np.sqrt(self.hat_f))*self.scale, 
             color = 'red', 
             linewidth=1, linestyle = '--')
    
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
    
    sb.histplot(data = self.data._latent_resid, 
                stat = 'density', 
                bins = 50, 
                facecolor=(0, 0, 0, 0),
                linewidth=1,
                edgecolor='black', 
                ax = ax6);
    
    ax6.set_xlabel('value residuals')
    ax6.set_ylabel('density')
    
    plt.tight_layout()
    plt.show()






