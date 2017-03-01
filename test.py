import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

unrate = pd.read_csv("UNRATE.csv", index_col=0)
unrate = unrate.iloc[-120:]
# unrate.UNRATE = np.log(unrate.UNRATE)
unrate.index = unrate.index.to_datetime()

train_frac = len(unrate) * 99//100
pred_range = unrate.index[train_frac - 4:]

model = sm.tsa.ARMA(unrate.iloc[:train_frac], (1,0))
result = model.fit()


plt.plot(unrate, 'b-', label='Unemployemnt Rate')
plt.plot(result.fittedvalues, 'r--', label='Model')
plt.plot(pred_range, result.predict(train_frac - 3, len(unrate), dynamic=True), 'g--', label='Prediction')

plt.show()

