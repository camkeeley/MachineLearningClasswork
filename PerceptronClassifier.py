# -*- coding: utf-8 -*-
"""
Created on Mon May  3 20:48:34 2021

@author: camke
"""

import numpy as np
from sklearn import datasets

# Perceptron Class
class Perceptron():
    def __init__(self, eta = 0.05, n_iter = 1000, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
                
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
    
    
    
    
# Now run the perceptron code on the iris data set   
irisData = datasets.load_iris()
perceptron_1 = Perceptron()
perceptron_2 = Perceptron()
perceptron_3 = Perceptron()


y1 = np.where(irisData.target == 0, 1, -1)

y2 = np.where(irisData.target == 1, 1, -1)

y3 = np.where(irisData.target == 2, 1, -1)

perceptron_1.fit(irisData.data[:,:], y1)


    
perceptron_2.fit(irisData.data[:,:], y2)


perceptron_3.fit(irisData.data[:,:], y3)



predictions = []


for i in range( len( irisData.data[:]) ):
    highestConfidence = max(perceptron_1.net_input(irisData.data[i]), perceptron_2.net_input(irisData.data[i]), perceptron_3.net_input(irisData.data[i]))
    if perceptron_1.predict(irisData.data[i]) == 1:
        predictions.append(0)
        
    elif perceptron_3.predict(irisData.data[i]) == 1:
        predictions.append(2)
        
    else:
        predictions.append(1)
""" 

for i in range( len( irisData.data[:]) ):
    if perceptron_1.net_input(irisData.data[i]) > perceptron_2.net_input(irisData.data[i]):
        predictions.append(0)
        
    elif perceptron_2.predict(irisData.data[i]) == 1:
        predictions.append(1)
        
    else:
        predictions.append(2)
"""
errorSum = 0
for i in range( len( irisData.data[:]) ):
    if irisData.target[i] != predictions[i]:
        errorSum += 1
        
print('\n Error Rate: ' + str(errorSum/len( irisData.data[:])) + '.') 


    
    
    
    
    
    
    
    
    
    
    
    
"""
y2 = []
for i in range(0, 50):
    y2.append(-1)
    
for i in range(50, 100):
    y2.append(1)
    
for i in range(100, 150):
    y2.append(-1)
"""

"""
y2 = []
for i in range(0, 100):
    y2.append(-1)
    
for i in range(100, 150):
    y2.append(1)
"""
"""
y2 = []
for i in range(0, 50):
    y2.append(-1)
    
for i in range(50, 100):
    y2.append(1)
"""
"""

y2 = []

for i in range(50, 100):
    y2.append(1)
    
for i in range(100, 150):
    y2.append(-1)
    
perceptron_1.fit(irisData.data[:100,:], y1)
p2X = irisData.data[0:50,:]
p2X.append(irisData.data[100:,:])
perceptron_2.fit(p2X, y2)
"""




