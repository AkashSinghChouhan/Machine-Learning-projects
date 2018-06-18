from sklearn.datasets import load_iris

iris=load_iris()
X=iris.data
y=iris.target

from sklearn.neighbors import KNeighborsClassifier


from sklearn.model_selection import GridSearchCV

k_range= range(1,31)
knn = KNeighborsClassifier() 
param_grid = dict(n_neighbors= k_range)


grid= GridSearchCV(knn, param_grid, cv=10, scoring ='accuracy')
grid.fit(X,y)

#use cv_results_ in place of grid_scores_
print(grid.grid_scores_)

#Check individual tuple
#print(grid.grid_scores_[0].parameters)
#print(grid.grid_scores_[0].cv_validation_scores)
#print(grid.grid_scores_[0].mean_validation_score)

grid_mean_scores =[result.mean_validation_score for result in grid.grid_scores_]
print(grid_mean_scores)

import matplotlib.pyplot as plt
plt.plot(k_range,grid_mean_scores)
plt.xlabel('Values of K')
plt.ylabel('Cross-validated accuracy score')
plt.show()

print("Best Score  :"+ str(grid.best_score_))
print("Best Parameter  :"+ str(grid.best_params_))
print("Best Estimator :"+ str(grid.best_estimator_))




# GRID SEARCH FOR MULTI PARAMETERS TUNING
k_range = range(1,31)
weight_options = ['uniform','distance']
param_grid = dict(n_neighbors = k_range, weights = weight_options)

grid= GridSearchCV(knn, param_grid , cv=10 ,scoring = 'accuracy')
grid.fit(X,y)

print("\n\nResults for multiple parameters using grid search")

print(grid.grid_scores_)
print("Best Score  :"+ str(grid.best_score_))
print("Best Parameter  :"+ str(grid.best_params_))
print("Best Estimator :"+ str(grid.best_estimator_))

#grid object will capture the most accurate model automatically so we can directly use it to make predictions instead of fitting the knn object with best tuned parameters
#grid.predict([[3,4,5,2]])

#Randomized GRID SEARCH FOR MULTI PARAMETERS TUNING

from sklearn.model_selection import RandomizedSearchCV

param_dist = dict(n_neighbors = k_range, weights = weight_options)
#for regression models we'll require continuous parameter, here we have discret parameter

rand =RandomizedSearchCV(knn , param_dist , cv = 10 , scoring ='accuracy',n_iter=10 , random_state=5)
rand.fit(X,y)

print("\n\nResults for RANDOMIZED multiple parameters tuning using randomized search")
print(rand.grid_scores_)
print("Best Score  :"+ str(rand.best_score_))
print("Best Parameter  :"+ str(rand.best_params_))
print("Best Estimator :"+ str(rand.best_estimator_))

print("\n\nRunning randomized search 20 times using for loop for better results")

scores=[]
for k in range(21):
    rand =RandomizedSearchCV(knn , param_dist , cv = 10 , scoring ='accuracy',n_iter=10 )
    rand.fit(X,y)
    scores.append(round(rand.best_score_,3))

print("\nBest scores =" +str(scores))


