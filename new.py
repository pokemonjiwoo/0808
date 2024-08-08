import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data.describe().to_csv('./new/pima-indians-diabetes.data.csv')


corr = (data.corr(method='pearson'))

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr, cmap = 'coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks=np.arange(0, 9, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(header)
ax.set_yticklabels(header)
plt.show()