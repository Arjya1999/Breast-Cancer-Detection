import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
import matplotlib.pyplot as plt
df_cancer= pd.DataFrame(np.c_[cancer['data'],cancer['target']],columns=np.append(cancer['feature_names'],['target']))
plt.figure(figsize=(20,10))
sns.heatmap(df_cancer.corr(),annot=True)
