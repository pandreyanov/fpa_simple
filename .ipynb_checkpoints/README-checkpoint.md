### A package for the "Nonparametric inference on counterfactuals in sealed first-price auctions" paper by Pasha Andreyanov and Grigory Franguridi.
It contains a class that fits the auction data using a symmetric first-price auction model with either additive or multiplicative heterogeneity, and predicts latent valuations and counterfactuals.

The interface of the package consists of 4 steps.

- pass a dataframe with auctionid and bid column names
- pass covariate (continuous and discrete) column names and create bid residuals and fitted values
- fit the non-parametric model
- predict latent bids, and also expected total surplus, potential bidder surplus and revenue, as functions of exclusion level

### Arxiv and Github repository
https://arxiv.org/abs/2106.13856

https://github.com/pandreyanov/pashas_simple_fpa

### Sample code

Package can be installed via pip from terminal

```python
pip install simple_fpa
```

Import typical auction data

```python
from simple_fpa import Model
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from pylab import rcParams

rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Arial"],
    "figure.dpi": 200
})

df = pd.read_csv('../_data/haile_data_prepared.csv', index_col=0)
```

Residualize the bids, print summary

```python
model = Model(data = df, auctionid_columns = ['auctionid'], bid_column = 'actual_bid')
cont_covs = ['adv_value', 'hhi', 'volume_total_1']
disc_covs = ['year', 'forest']
model.residualize(cont_covs, disc_covs, 'multiplicative')

model.summary()

# # we can simulate artificial data, of course
# model.data._resid = np.sort(np.random.uniform(0,1,size = len(model.data)))
```

Trim the residuals, fit the model and predict latent valuations

```python
model.trim_residuals(10)
model.fit(smoothing_rate = 0.2, trim_percent = 5, reflect = True)
model.predict()
```

Make confidence intervals and confidence bands

```python
model.make_ci(99, hyp = 'twosided')
model.make_cb(99, draws = 1000, hyp = 'twosided')
model.make_cicb_for_ts(99, draws = 1000, hyp = 'twosided')
model.plot_stats()

```

Find optimal exclusion level and plot counterfactuals

```python
model.find_optimal_u()
model.plot_counterfactuals()
```

Inspect the data

```python
model.data.sample(5)
```

### Predictions

The counterfactuals are populated into the original dataset, ordered by the magnitude of bid redisuals. Some observations will not have a prediction, as they will be ignored (trimmed) in the non-parametric estimation. I use underscore in front of all variables created by the package.

- *_resid* : bid residuals
- *_fitted* : bid fitted values
- *_trimmed* : variable takes 1 if observations were omitted (trimmed) and 0 otherwise
- *_u* : u-quantile levels, takes values between 0 and 1

- *_hat_q* : estimate of quantile density of bid residuals
- *_hat_v* : estimate of quantile function of value residuals

- *_latent_resid* : same as *_hat_v*

- *_hat_ts* : total surplus as function of exclusion level u
- *_hat_bs* : (one) potential bidder surplus as function of exclusion level u
- *_hat_rev* : auctioneer revenue as function of exclusion level u

- *q_ci*, *v_ci*, *_bs_ci*, *_ts_ci*, *_rev_ci* : simulated confidence intervals
- *q_cb*, *v_cb*, *_bs_cb*, *_ts_cb*, *_rev_cb* : simulated confidence bands

- *q_cb_asy*, *v_cb_asy*, *_bs_cb_asy*, *_rev_cb_asy* : asymptotic (theoretical) confidence intervals