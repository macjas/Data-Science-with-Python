#############################################################################################################
# 2. Kernel Density Estimation Plots
#############################################################################################################
# The normal imports
import numpy as np
from numpy.random import randn
import pandas as pd
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#Create dataset
dataset = randn(25)
#Create rugplot
sns.rugplot(dataset)
sns.kdeplot(dataset, shade=True, kernel='gau')
plt.hist(dataset, normed=True, color="#6495ED", alpha=.5)



