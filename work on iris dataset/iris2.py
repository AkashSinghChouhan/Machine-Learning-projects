from sklearn.datasets import load_iris
iris= load_iris()
X= iris.data
y=iris.target

from sklearn.neighbors import KNeighborsClassifier
knn5= KNeighborsClassifier(n_neighbors=5)
knn5.fit(X,y)

from sklearn.linear_model import  LogisticRegression
logreg=  LogisticRegression()
logreg.fit(X,y)

from sklearn import metrics
k5=knn5.predict(X)
lr=logreg.predict(X)
print("Accuracy of knn5 = ")
print(metrics.accuracy_score(y,k5))
print("Accuracy of logreg = " )
print(metrics.accuracy_score(y,lr))

from sklearn.cross_validation import train_test_split
X_train ,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)

knn5= KNeighborsClassifier(n_neighbors=5)
logreg=  LogisticRegression()

knn5.fit(X_train,y_train)
logreg.fit(X_train,y_train)

k5=knn5.predict(X_test)
lr=logreg.predict(X_test)
print("Accuracies after splitting data....")

print("Accuracy of knn5 = " )
print(metrics.accuracy_score(y_test,k5))
print("Accuracy of logreg = " )
print(metrics.accuracy_score(y_test,lr))

#finding best value of k for knn

scores= []
r =range(1,26)
for i in r:
    knn= KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train,y_train)
    k=knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test,k))

import  matplotlib.pyplot as plt


plt.plot(r,scores)
plt.xlabel('Values of k')
plt.ylabel('Accuracy scores')
plt.show()




















