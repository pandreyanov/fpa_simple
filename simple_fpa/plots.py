import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sb

def plot_stats(self):
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2)
    sb.countplot(x = self.data.groupby(by = 'auctionid')._bidders.first().astype(int), 
                 facecolor=(0, 0, 0, 0),
                 linewidth=1,
                 edgecolor='black', 
                 ax = ax1)
    
    ax1.set_xlabel('bidders')
    
    sb.histplot(data = self.data._resid, 
                stat = 'density', 
                bins = 50, 
                facecolor=(0, 0, 0, 0),
                linewidth=1,
                edgecolor='black', 
                ax = ax2);
    
    ax2.plot(self.u_grid*self.scale+self.intercept, 
             self.hat_f, 
             color = 'red', 
             label = 'smooth $\hat f(b)$', 
             linewidth=1, 
             alpha = .7)
    
    ax2.set_xlabel('bid residuals')
    ax2.set_ylabel('density')
    ax2.legend()
    
    ax3.plot(self.u_grid, self.A_1, label = '$A_1$')
    ax3.plot(self.u_grid, self.A_2, label = '$A_2$')
    ax3.plot(self.u_grid, self.A_3, label = '$A_3$')
    ax3.plot(self.u_grid, self.A_4, label = '$A_4$')
    ax3.legend()
        
    ax4.plot(self.u_grid, self.hat_q, label = 'smooth $\hat q(u)$')
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






