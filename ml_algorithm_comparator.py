# -*- coding: utf-8 -*-
"""
Created on Tue May 18 00:01:23 2021

@author: camke
"""


from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
import numpy as np

digits = datasets.load_digits()

n_samples = len(digits.images)

data = digits.images.reshape(n_samples, -1)





News_X, Newsy = datasets.fetch_20newsgroups_vectorized(subset = 'all', return_X_y=True )

svd = TruncatedSVD(n_components = 100)
#News_X = svd.fit_transform(News_X)






clf_1 = svm.SVC(gamma = 0.01, C = 10)

clf_2 = MLPClassifier(solver = 'lbfgs', alpha = 0.02, random_state = 1, max_iter = 100, 
                          early_stopping = True, hidden_layer_sizes=[200,500])
clf_3 = svm.SVC(gamma = 0.01, C = 10)

clf_4 = svm.SVC(gamma = 0.01, C = 10)

clf_5 = MLPClassifier(solver = 'adam', alpha = 0.000000001, random_state = 1, max_iter = 200, 
                          early_stopping = True, hidden_layer_sizes= (100,))
clf_6 = RandomForestClassifier(random_state=0)

clf_7 = DecisionTreeClassifier(random_state=0)
    
clf_8 = LogisticRegression(random_state=1)


X_std = np.copy(data[:,:])
for i in range( 0,64 ):
    if data[:,i].std() != 0:
        X_std[:,i] = (data[:,i] - data[:,i].mean()) / data[:,i].std()
"""
scores = cross_val_score(clf_6, News_X, Newsy, cv = 10)
print("%0.2f accuracy of news Classifier with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


scores = cross_val_score(clf_1, X_std, digits.target, cv = 10)
print("%0.2f accuracy of digits classifier with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
"""


X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size = 0.1)

ml_pipeline = make_pipeline(StandardScaler(), clf_1)

ml_pipeline.fit(X_train, y_train)

predictions2 = ml_pipeline.predict(X_test)




X_train_news, X_test_news, y_train_news, y_test_news = train_test_split(News_X, Newsy, test_size = 0.1)

clf_6.fit(X_train_news, y_train_news)

predictions = clf_6.predict(X_test_news)




   
 
sumCorrect_2 = 0
for i in range( len(y_test_news)):
    if predictions[i] == y_test_news[i]:
        sumCorrect_2+= 1
print('Overall Accuracy Rate For News Dataset using Random Forest Classifier: ' + str(sumCorrect_2 / len(predictions)))


sumCorrect = 0
for i in range( len(y_test)):
    if predictions2[i] == y_test[i]:
        sumCorrect+= 1
 
scaler = StandardScaler()
scaler.fit(data)
X_std = scaler.transform(data)
print('Overall Accuracy Rate For Digits Dataset using SVM: ' + str(sumCorrect / len(predictions2)))


disp = metrics.plot_confusion_matrix(ml_pipeline, X_test, y_test)