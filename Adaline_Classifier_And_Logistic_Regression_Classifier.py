# -*- coding: utf-8 -*-
"""
Created on Wed May  5 15:34:42 2021

@author: camke
"""
import numpy as np
from sklearn import datasets

class Adaline():
    def __init__(self, eta = 0.001, n_iter = 100, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
            
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        return X
    
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
    
adaline_1 = Adaline()
adaline_2 = Adaline()
adaline_3 = Adaline()

iris_Data = datasets.load_iris()

X_std = np.copy(iris_Data.data[:,:])
        
for i in range( 0,4 ):
    X_std[:,i] = (iris_Data.data[:,i] - iris_Data.data[:,i].mean()) / iris_Data.data[:,i].std()



y1 = np.where(iris_Data.target == 0, 1, -1)
adaline_1.fit(X_std, y1)

y2 = np.where(iris_Data.target == 1, 1, -1)
adaline_2.fit(X_std, y2)


y3 = np.where(iris_Data.target == 2, 1, -1)
adaline_3.fit(X_std, y3)


predictions = []



for i in range( 0, 150):
    irisClassifierOne = adaline_1.net_input(X_std[i])
    irisClassifierTwo = adaline_2.net_input(X_std[i])
    irisClassifierThree = adaline_3.net_input(X_std[i])
    
    if irisClassifierOne > irisClassifierTwo and irisClassifierOne  > irisClassifierThree:
        predictions.append(0)
        
    

        
    elif irisClassifierTwo > irisClassifierOne and irisClassifierTwo > irisClassifierThree:
        predictions.append(1)

                
           
    else:
        predictions.append(2)
        
"""    print(str(irisClassifierOne) + ', ' + str(irisClassifierTwo) + ', ' + str(irisClassifierThree))
"""
x = 0
for i in range( len( iris_Data.data[:]) ):
    if iris_Data.target[i] != predictions[i]:
        x += 1
print('Iris Dataset Error Rate using Adaline:' + str((x)/len( iris_Data.data[:])))




wine_Data = datasets.load_wine()

W_std = np.copy(wine_Data.data[:,:])
        
for i in range( 0, 13 ):
    W_std[:,i] = (wine_Data.data[:,i] - wine_Data.data[:,i].mean()) / wine_Data.data[:,i].std()

adaline_4 = Adaline(eta = 0.001)
adaline_5 = Adaline(eta = 0.001)
adaline_6 = Adaline(eta = 0.001)

w1 = np.where(wine_Data.target == 0, 1, -1)
adaline_4.fit(W_std, w1)

w2 = np.where(wine_Data.target == 1, 1, -1)
adaline_5.fit(W_std, w2)

w3 = np.where(wine_Data.target == 2, 1, -1)
adaline_6.fit(W_std, w3)


predictions_wine = []
for i in range( len( wine_Data.target ) ):
    
    wineClassifierOne = adaline_4.net_input(W_std[i])
    wineClassifierTwo = adaline_5.net_input(W_std[i])
    wineClassifierThree = adaline_6.net_input(W_std[i])

    if wineClassifierOne > wineClassifierTwo and wineClassifierOne > wineClassifierThree:
        predictions_wine.append(0)
    
    elif wineClassifierTwo > wineClassifierOne and wineClassifierTwo > wineClassifierThree:
        predictions_wine.append(1)
        
    else:
        predictions_wine.append(2)
        
w = 0
for i in range( len( wine_Data.data[:]) ):
    if wine_Data.target[i] != predictions_wine[i]:
        w += 1
print('\nWine Dataset Error Rate using Adaline:' + str((w)/len( wine_Data.data[:])))



class LogisticRegression():
    def __init__(self, eta = 0.05, n_iter = 1000, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
            
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = ( -y.dot(np.log(output)) - ((1 - y).dot(np.log(1-output))) )
            self.cost_.append(cost)
        
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250 ) ) ) 
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)
    
    
log_reg_1 = LogisticRegression()
log_reg_2 = LogisticRegression()
log_reg_3 = LogisticRegression()

log_reg_X_std = np.copy(iris_Data.data[:,:])
        
for i in range( 0,4 ):
    log_reg_X_std[:,i] = (iris_Data.data[:,i] - iris_Data.data[:,i].mean()) / iris_Data.data[:,i].std()



log_Reg_y1 = np.where(iris_Data.target == 0, 1, 0)
log_reg_1.fit(log_reg_X_std, log_Reg_y1)

log_Reg_y2 = np.where(iris_Data.target == 1, 1, 0)
log_reg_2.fit(log_reg_X_std, log_Reg_y2)


log_Reg_y3 = np.where(iris_Data.target == 2, 1, 0)
log_reg_3.fit(log_reg_X_std, log_Reg_y3)


predictions_log_Reg_Iris = []



for i in range( 0, 150):
    lr_irisClassifierOne = log_reg_1.activation(log_reg_1.net_input(log_reg_X_std[i]))
    lr_irisClassifierTwo = log_reg_2.activation(log_reg_2.net_input(log_reg_X_std[i]))
    lr_irisClassifierThree = log_reg_3.activation(log_reg_3.net_input(log_reg_X_std[i]))
    
    if lr_irisClassifierOne > lr_irisClassifierTwo and lr_irisClassifierOne  > lr_irisClassifierThree:
        predictions_log_Reg_Iris.append(0)
        
    

        
    elif lr_irisClassifierTwo > lr_irisClassifierOne and lr_irisClassifierTwo > lr_irisClassifierThree:
        predictions_log_Reg_Iris.append(1)

                
           
    else:
        predictions_log_Reg_Iris.append(2)
        
"""    print(str(lr_irisClassifierOne) + ', ' + str(lr_irisClassifierTwo) + ', ' + str(lr_irisClassifierThree))
"""
log_Reg_Iris_Errors = 0
for i in range( len( iris_Data.data[:]) ):
    if iris_Data.target[i] != predictions_log_Reg_Iris[i]:
        log_Reg_Iris_Errors += 1
print('\nIris Dataset Error Rate using Logistic Regression:' + str((log_Reg_Iris_Errors)/len( iris_Data.data[:])))





log_Reg_W_std = np.copy(wine_Data.data[:,:])
        
for i in range( 0, 13 ):
    log_Reg_W_std[:,i] = (wine_Data.data[:,i] - wine_Data.data[:,i].mean()) / wine_Data.data[:,i].std()

log_reg_4 = LogisticRegression(eta = 0.005)
log_reg_5 = LogisticRegression(eta = 0.005)
log_reg_6 = LogisticRegression(eta = 0.005)

log_reg_w1 = np.where(wine_Data.target == 0, 1, 0)
log_reg_4.fit(log_Reg_W_std, log_reg_w1)

log_reg_w2 = np.where(wine_Data.target == 1, 1, 0)
log_reg_5.fit(log_Reg_W_std, log_reg_w2)

log_reg_w3 = np.where(wine_Data.target == 2, 1, 0)
log_reg_6.fit(log_Reg_W_std, log_reg_w3)


log_Reg_predictions_wine = []
for i in range( len( wine_Data.target ) ):
    
    wineClassifierOne = log_reg_4.activation(log_reg_4.net_input(log_Reg_W_std[i]))
    wineClassifierTwo = log_reg_5.activation(log_reg_5.net_input(log_Reg_W_std[i]))
    wineClassifierThree = log_reg_6.activation(log_reg_6.net_input(log_Reg_W_std[i]))

    if wineClassifierOne > wineClassifierTwo and wineClassifierOne > wineClassifierThree:
        log_Reg_predictions_wine.append(0)
    
    elif wineClassifierTwo > wineClassifierOne and wineClassifierTwo > wineClassifierThree:
        log_Reg_predictions_wine.append(1)
        
    else:
        log_Reg_predictions_wine.append(2)
        
"""    print(str(wineClassifierOne) + ', ' + str(wineClassifierTwo) + ', ' + str(wineClassifierThree))
"""
        
log_Reg_Errors_w = 0
for i in range( len( wine_Data.data[:]) ):
    if wine_Data.target[i] != log_Reg_predictions_wine[i]:
        log_Reg_Errors_w += 1
print('\nWine Dataset Error Rate using Logistic Regresion:' + str((log_Reg_Errors_w)/len( wine_Data.data[:])))
    
        