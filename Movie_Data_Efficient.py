# -*- coding: utf-8 -*-
"""
Created on THur May  13 01:20:54 2021

@author: camke
"""
from glob import glob
import numpy as np
import pandas as pd

class MovieData():
    def __init__(self, eta = 0.01, n_iter = 1, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
            
        rgen = np.random.RandomState(self.random_state)
        self.w_U_ = {}
        self.w_M_ = {}
        self.cost_ = []

        for key in X:
            jd = rgen.normal(loc = 0.0, scale = 0.01, size = 20)
            kd = key
            
        
            self.w_U_[kd] = {}
            self.w_U_[kd] = jd
            
            
            
            
            #self.w_U_ = np.append(rgen.normal(loc = 0.0, scale = 0.01, size = 5))
            
        for i in range( len( y ) ):
            jd = rgen.normal(loc = 0.0, scale = 0.01, size = 20)
            
            self.w_M_[i] = {}
            self.w_M_[i] = jd


        
        
        
        
        for i in range(self.n_iter):
           
            for key in self.w_M_:
                self.w_M_[key] = self.w_M_[key]  - self.eta * self.movieUpdate(self.w_M_[key], self.w_U_, y, X, key )
                
            for key in self.w_U_:
                """
                print('\n????' + str(self.eta * self.userUpdate(self.w_M_, each, y, X )))
                """
                self.w_U_[key] = self.w_U_[key] - self.eta * self.userUpdate(self.w_M_, self.w_U_[key], y, X, key )
                
            
            self.eta = self.eta * 0.9
            
#      self.cost_.append(cost)

 
#     cost = (errors**2).sum() / 2.0

        print('Done Fitting.')
        return self    


    def movieUpdate(self, movie_weights, user_weights, movies, users, movie_ID):
        #For each user that rated the movie, sum the costs of the predictions 
        #of each user. 
        #Need: user id's, user ratings, user weight vectors
        sumOfCosts = [0] * 20
        """
        print("\n### " + str(movie_weights[0]))
        """
        for index, each in movies[movie_ID].iterrows():
            true_Rating = each['Rating_Score']
            prediction = self.predict(users[each['User_ID']], movies[movie_ID], user_weights[each['User_ID']], movie_weights)
            sumOfCosts += (prediction - true_Rating) * user_weights[each['User_ID']] + .001 * movie_weights
            print('\n MOVIE prediction: ' + str(prediction) + '\ntrue score: ' + str(true_Rating))

          
        if len( movies[movie_ID] ) > 0:
                    return ( 1 / len( movies[movie_ID] ) * sumOfCosts) 

        else:
            return 0
        
    
    def userUpdate(self, movie_weights, user_weights, movies, users, user_ID):
        #For each movie that the user rated, sum the costs of the predictions 
        #of each rating. 
        #Need: movie id's, movie ratings, movie weight vectors
        sumOfCosts = [0] * 20
        
        for key in users[user_ID]:
            
            true_Rating = users[user_ID][key]
           
            prediction = self.predict(users[user_ID], movies[key], user_weights, movie_weights[key])
            first_p = (prediction - true_Rating) * movie_weights[key] 
            second_p =  0.001 * user_weights
            sumOfCosts =  np.add(np.add(first_p, second_p), sumOfCosts)

            """
            print('\n!!!!!!!!!!' + str(user_weights[1:]) + '\n' + str(movie_weights[key][1:]))
            """
            print('\n USER prediction: ' + str(prediction) + '\ntrue score: ' + str(true_Rating))
            

        return (( 1 / len( users[user_ID] )) * sumOfCosts)
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        return X
    
    def predict(self, u, M, user_weights, movie_weights):
        return self.activation(np.dot(user_weights, movie_weights) + (self.averageMovieScore(M) + self.averageUserScore(u))/ 2)
    
    def averageMovieScore(self, X):
        return X['Rating_Score'].sum() / len(X)
        
    def averageUserScore(self, X):
        return sum(list(X.values())) / len(X) 
"""    def _update_weights(self, xi, target):
"""    
    
#filenames = ('mv_0000001.txt', 'mv_0000002.txt', 'mv_0000003.txt', 'mv_0000004.txt', 'mv_0000005.txt')
filenames = glob('mv*.txt')
#movieFiles = [pd.read_csv(f, names = ['User_ID', 'Rating_Score', 'Date'], usecols=(['User_ID', 'Rating_Score']), skiprows = 11, squeeze = True).to_dict() for f in filenames]

movieFiles = [pd.read_csv(f, names = ['User_ID', 'Rating_Score', 'Date'], usecols=(['User_ID', 'Rating_Score']), skiprows = 11) for f in filenames]
movieFiles.sort(key = len, reverse = True)
movieFiles = movieFiles[:100]
"""
print (str(movieFiles[0].User_ID))
print ('\nNumber of Ratings for this movie:' + str(len(movieFiles[0])))
print ('Average Score for this movie:' + str(movieFiles[0].Rating_Score.mean()))
"""
#testFiles = [pd.read_csv(f, names = ['User_ID', 'Rating_Score', 'Date'], usecols=(['User_ID', 'Rating_Score']), nrows = 11, skiprows = 1, squeeze = True).to_dict() for f in filenames]

testFiles = [pd.read_csv(f, names = ['User_ID', 'Rating_Score', 'Date'], usecols=(['User_ID', 'Rating_Score']), nrows = 11, skiprows = 1) for f in filenames]
testFiles.sort(key = len, reverse = True)
testFiles = testFiles[:100]
print('Done soring.')
ratingSum = 0
users = {}

for index, each in enumerate( movieFiles ):
    ratingSum += each.Rating_Score.mean()
    
    for User_ID_index, x in enumerate (each['User_ID']):        
        r_Score = each.loc[ int(User_ID_index), 'Rating_Score' ]       
        
        if x not in users:
            
            users[x] = {}
    
        users[x][index] = r_Score


print('Done building users.')    




"""
print ('Average Score for all movies:' + str(ratingSum/len(movieFiles)))
"""

movData =  MovieData()
movData.fit(users, movieFiles )

print('Done training data.')
     
RMSE = 0
sampleSize = 0
for index, each in enumerate(testFiles):
    
    for User_ID_index, x in each.iterrows():        
        if x['User_ID'] in users:
            test_user = users[x['User_ID']]
            test_Score = x['Rating_Score']
        
            """
            weight_Index = 0
            for i in range( len(movData.w_U_) ):
                if movData.w_U_[i][0] == x['User_ID']:
                    weight_Index = i
                    break
            """
                
            test_user_weights = movData.w_U_[x['User_ID']]
            test_movie_weights = movData.w_M_[index]
        
            prediction =  movData.predict(test_user, testFiles[index], test_user_weights, test_movie_weights)

            RMSE += (test_Score - prediction) ** 2
            
            sampleSize += 1
            """
            print('$')
            """
        """     
        print('%')
        """
RMSE = np.sqrt(RMSE/sampleSize)
print ('RMSE: ' + str(RMSE) + '.')       
print ('Assignment not fully completed, only sorts 1000 movies to find the 100 most rated movies because it takes to long to the full sample size, and does not shuffle data.')       

