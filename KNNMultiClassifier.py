# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 21:39:40 2021

@author: camke
"""

import numpy  as np
from sklearn import datasets


class KNNMultiClassifier:
    def __init__( self, nNeighbors):
        self.knn = nNeighbors
        self.X = []
        self.y = []
        
    def fit( self, X, y ):
        self.X = X
        self.y = y
        return self
    
    def predict ( self, x ):
        dist = []
        for i in range( len( self.X ) ):
            dist.append( (np.linalg.norm( x - self.X[i,:] ), self.y[i] ) )
            
            
        dist = sorted(dist, key=lambda l:l[0])
            
        zero_score = 0
        one_score = 0
        two_score = 0

        for i in range(0, self.knn - 1):
            if dist[i][1] == 0:
                zero_score+=1
                
            else:
                zero_score-=1
                
            if dist[i][1] == 1:
                one_score+=1
                
            else:
                one_score-=1
                
            if dist[i][1] == 2:
                two_score+=1
                
            else:
                two_score-=1
                
        highest_probability_class = max(zero_score, one_score, two_score)
                

      
        if highest_probability_class == zero_score:
            return 0
        elif highest_probability_class == one_score:
            return 1
        else:
            return 2
  
    def predictRegression ( self, x ):
        dist = []
        for i in range( len( self.X ) ):
            dist.append( (np.linalg.norm( x - self.X[i,:] ), self.y[i] ) )
            
            
        dist = sorted(dist, key=lambda l:l[0])
       
        priceAverage = 0
        for i in range(0, self.knn - 1):
            priceAverage += dist[i][1]
        
        priceAverage = priceAverage/self.knn
        
        return priceAverage
            

  
irisData = datasets.load_iris()
wineData = datasets.load_wine()
bostonData = datasets.load_boston()


knn_iris = KNNMultiClassifier(5)

knn_iris.fit(irisData.data[:,:], irisData.target)
irisPredictedLabels = []
for i in range( len( irisData.data[:] ) ):
    irisPredictedLabels.append( knn_iris.predict( irisData.data[i,:] ) )

iris_correctPredictions = 0
for i in range (len( irisData.data[:] ) ):
    if irisPredictedLabels[i] == irisData.target[i]:
        iris_correctPredictions+=1
        
print('Iris Prediction Error Rate: ' + str(1 - iris_correctPredictions/150) + '.') 
print('Iris Prediction Accuracy: ' + str(iris_correctPredictions/150) + '.') 

        
    
knn_wine = KNNMultiClassifier(5)

knn_wine.fit(wineData.data[:,:], wineData.target)
winePredictedLabels = []
for i in range( len( wineData.data[:] ) ):
    winePredictedLabels.append( knn_wine.predict( wineData.data[i,:] ) )
    
wine_correctPredictions = 0
for i in range (len( wineData.data[:] ) ):
    if winePredictedLabels[i] == wineData.target[i]:
        wine_correctPredictions+=1
        
print('\nWine Prediction Error Rate: ' + str(1 - wine_correctPredictions/150) + '.') 
print('Wine Prediction Accuracy: ' + str(wine_correctPredictions/150) + '.') 



knn_Boston = KNNMultiClassifier(5)

knn_Boston.fit(bostonData.data[0:399,:], bostonData.target)
bostonPredictedLabels = []
for i in range( 0, 399 ):
    bostonPredictedLabels.append( knn_Boston.predictRegression( bostonData.data[i,:] ) )
  
    
trained_Data = bostonData
for i in range( 0, len(bostonPredictedLabels[:])):
    trained_Data.target[i] = bostonPredictedLabels[i]
    
knn_TrainedBostonData = KNNMultiClassifier(5)
knn_TrainedBostonData.fit(trained_Data.data[:399,:], trained_Data.target[:399])

bostonTestLabels = []
RMSE = 0
sumOfPricesDifferencesSquared = 0
for i in range ( 400, len( bostonData.data[:] ) ):
    predictionPrice = knn_TrainedBostonData.predictRegression( bostonData.data[i,:] ) 
    sumOfPricesDifferencesSquared += (bostonData.target[i] - predictionPrice ) ** 2
    bostonTestLabels.append(predictionPrice)
    
averageOfPriceDifferencesSquared = sumOfPricesDifferencesSquared/( len( bostonData.data[:] ) - 400 )
RMSE = np.sqrt(averageOfPriceDifferencesSquared)  

print('\nRMSE: ' + str(RMSE) + '.') 




