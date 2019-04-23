
ECON 9000_ps1 (Machine Learning)                                     Krishna Khatri

Q1) Web Scraping 
 
Part 1) Request:
Import urllib.request, os and time packages. 
Create ‘html_files’ folder if it doesn’t exist in directory.
Send request using urlib.request.urlopen command.
Read response and write to html.
Sleep until 20 seconds.



Part 2) Parse:
Install packages like BeautifulSoup, pandas, os and glob.
Create folder named “parsed_files”, if it doesn’t exist.
Using soup.find function, make html readable.
Using find_all or find command, find data on Geek_Rating, Avg_Rating, Vote and Title from appropriate place.
Append and save all parsed data to data.csv file by category. 

Q2) Machine Learning

Part 1) Linear Regression 

I have collected data by web scraping on Vote, Greek_Rating and Avg_Rating of different type of board games. My interest lies in identifying whether Greek_Rating and Avg_Rating help us to predict vote received by particular board game. I am using linear regression model and have imported statsmodels package. Data from first 200 types of board games are used for this machine learning analysis.

Import following package in text editor:

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

I found that Geek_Rating is positively and Avg_Rating is negatively related to Vote. One point increases in Geek_Rating leads to increase in vote number by 12610 and one point increase in Avg_Rating actually leads to reduction in Vote number by 4324. 
 
Using OLS coefficients and mean value of independent variables, I predict the Vote received by particular board game using Geek_Rating of board game and Avg_Rating of board game. 

Part 2) Classification using sklearn

Here, I am interested in classifying games by either receiving enough vote or receiving not enough vote based on their vote count. I am using data on vote, Geek_Rating and Avg_Rating of first 200 board games. Game with highest number of vote is 83000 and game with lowest number of vote is 30. So I need to rescale it. 

Import following commands in text editor:
    
                       
	import seaborn as sns
	import numpy as np
        import pandas as pd
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

Read the data using pd.read_csv command and save it to df.  Cut off number of vote is set to be 10000 so anything less than that implies game with not enough vote and game with more than 10000 vote are classified as game with enough vote. After using labelencoder and label_ratings.fit_transformation command to Vote, I plot figure of enough vote and not enough vote. 
Then training set is set as X_train, y_train and test set is set as X_test and y_test. After scaling data, I use 3 different types of classifier to train and test using prediction. I summarize that in confusion matrix table. 

Classifiers that I am using are RandomForestClassifier(rfc), svm.SVC and MLPClassifier(mlpc). Comparison between confusion matrix of above mentioned classifiers shows that  MLPClassifier is better in predicting than other two classifiers in current situations.
