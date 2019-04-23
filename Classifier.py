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

df = pd.read_csv("data1.csv") 
print(df.head()) 
print(df.info())
print(df.describe(include=[np.number])) 

bins=(25, 10000, 83000) 
group_names=['Not Enough', 'Enough']
df['Vote']=pd.cut(df['Vote'], bins=bins, labels=group_names)
print(df['Vote'].unique()) 

label_ratings=LabelEncoder()
df['Vote'] = label_ratings.fit_transform(df['Vote'])
print(df.head()) 
print(df['Vote'].value_counts()) 
plt.savefig('fig.png')


X=df[['Avg_Rating','Geek_Rating']]
y=df['Vote']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

print(X_train[:10])

rfc=RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc=rfc.predict(X_test) 

print(classification_report(y_test, pred_rfc))
print(confusion_matrix(y_test, pred_rfc)) 

clf=svm.SVC()(n_estimators=100)
clf.fit(X_train, y_train)
pred_clf=clf.predict(X_test)
print(classification_report(y_test, pred_clf))
print(confusion_matrix(y_test, pred_clf)) 

mlpc=MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=500)
mlpc.fit(X_train,y_train)
pred_mlpc=mlpc.predict(X_test)
print(classification_report(y_test, pred_mlpc))
print(confusion_matrix(y_test, pred_mlpc))

