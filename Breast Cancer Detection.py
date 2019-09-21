import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
df_cancer= pd.DataFrame(np.c_[cancer['data'],cancer['target']],columns=np.append(cancer['feature_names'],['target']))
#plt.figure(figsize=(20,10))
#sns.heatmap(df_cancer.corr(),annot=True)
X = df_cancer.iloc[:, :-1].values
y = df_cancer.iloc[:, 30].values
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(X)
test_set_scaled = np.reshape(y, (y.shape[0],1))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(training_set_scaled,test_set_scaled, test_size = 0.20, random_state=5)
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
svc_model = SVC()
svc_model.fit(X_train, y_train)
from sklearn.model_selection import GridSearchCV
parameters = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
grid_search = GridSearchCV(estimator = svc_model,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1,verbose=4)
grid_search.fit(X_train,y_train)
best_parameter = grid_search.best_params_
grid_predictions = grid_search.predict(X_test)
cm = confusion_matrix(y_test, grid_predictions)
print(classification_report(y_test,grid_predictions))