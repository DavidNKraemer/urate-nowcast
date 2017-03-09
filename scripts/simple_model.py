import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('talk')
sns.set_style('white')

unrate = pd.read_csv("../data/UNRATE.csv", index_col=0)

tslice = 40 
test = 8 

p, q, r = 4, 1, 0

unrate = unrate.iloc[-tslice:]
unrate.index = unrate.index.to_datetime()

pred_range = unrate.index[-test-1:]

model = sm.tsa.ARIMA(unrate.diff().iloc[1:-test], (p,q,r))
result = model.fit()

conf_int = np.empty((test+1,2))
conf_int[0] = (0,0)
forecast, stderr, conf_int[1:,:] = result.forecast(test)

simulated = np.empty(test+1)
simulated[0] = unrate.iloc[-test-1].values
simulated[1:] = unrate.iloc[-test].values + forecast.cumsum()

fig, ax = plt.subplots(1, 1)

ax.plot(unrate, 'k-', label='Unemployemnt Rate')
ax.plot(pred_range, simulated, 'b', label='Prediction (with 95% CI)')
ax.plot(pred_range, simulated + conf_int[:,0], 'b--')
ax.plot(pred_range, simulated + conf_int[:,1], 'b--')

ax.fill_between(pred_range, simulated, simulated + conf_int[:,0],
        where=simulated + conf_int[:,0] <= simulated, facecolor='blue',
        interpolate=True, alpha=0.1)

ax.fill_between(pred_range, simulated, simulated + conf_int[:,1],
        where=simulated + conf_int[:,1] >= simulated, facecolor='blue',
        interpolate=True, alpha=0.1)

ax.set_xlabel('Month')
ax.set_ylabel('Unemployment rate (National)')
ax.set_title('Unemployment rate with ARIMA({},{},{}) model prediction'.format(p,q,r))
ax.legend()
sns.despine()

plt.savefig('../figures/ur-arima({},{},{}).svg'.format(p,q,r), bbox_inches='tight')

