import pandas as pd 
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
import statsmodels.api as sm

df1 = pd.read_csv("data1.csv") 
df1.head()
plt.style.use('seaborn')
df1.plot(x='Geek_Rating', y='Vote', kind='scatter')
plt.show()

df1.plot(x='Avg_Rating', y='Vote', kind='scatter')
plt.show()
df1['const'] = 1
reg1 = sm.OLS(endog=df1['Vote'], exog=df1[['const','Geek_Rating', 'Avg_Rating']])
type(reg1)
results = reg1.fit()
type(results)
print(results.summary())
mean_Geek = np.mean(df1.Geek_Rating)
mean_Avg = np.mean(df1.Avg_Rating)
Predicted_Vote = (-1)* 43660 + 12610 * 6.582 - 4323.7958 * 7.2296

results.predict(exog=[1, mean_Geek,mean_Avg])

df1_plot = df1.dropna(subset=['Vote','Geek_Rating', 'Avg_Rating'])



plt.scatter(df1_plot['Geek_Rating'], results.predict(), alpha=0.5, label='predicted')


plt.scatter(df1_plot['Geek_Rating'], df1_plot['Vote'], alpha=0.5, label='observed')

plt.legend()
plt.title('OLS predicted values')
plt.xlabel('Geek_Rating')
plt.ylabel('Vote')
plt.show()
plt.scatter(df1_plot['Avg_Rating'], results.predict(), alpha=0.5, label='predicted')



plt.scatter(df1_plot['Avg_Rating'], df1_plot['Vote'], alpha=0.5, label='observed')

plt.legend()
plt.title('OLS predicted values')
plt.xlabel('Avg_Rating')
plt.ylabel('Vote')
plt.show()