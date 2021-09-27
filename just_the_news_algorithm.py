# -*- coding: utf-8 -*-
"""
Created on Tue May 18 03:42:46 2021

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

svd = TruncatedSVD(n_components = 100)

News_X, Newsy = datasets.fetch_20newsgroups_vectorized(subset = 'all', return_X_y=True )



News_X = svd.fit_transform(News_X)

X_train, X_test, y_train, y_test = train_test_split(News_X, Newsy, test_size = 0.1)


"""
scaler = MaxAbsScaler()
scaler.fit(News_X)
News_X = scaler.transform(News_X)
"""

clf_1 = svm.SVC(gamma = 1, C = 10)

clf_2 = MLPClassifier(solver = 'adam', alpha = 0.000000001, random_state = 1, max_iter = 200, 
                          early_stopping = True, hidden_layer_sizes= (100,))
clf_3 = RandomForestClassifier(random_state=0)

clf_4 = DecisionTreeClassifier(random_state=0)
    
clf_5 = LogisticRegression(random_state=1)


#grid = GridSearchCV( clf_2)
"""
scores = cross_val_score(clf_3, News_X, Newsy, cv = 10)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
"""
clf_1.fit(X_train, y_train)





predictions = clf_1.predict(X_test)
   
 
sumCorrect_2 = 0
for i in range( len(y_test)):
    if predictions[i] == y_test[i]:
        sumCorrect_2+= 1
print('Accuracy Rate For News using Random Forest Classifier: ' + str(sumCorrect_2 / len(predictions)))

disp = metrics.plot_confusion_matrix(clf_1, X_test, y_test)