import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,0,1],
                    [0,1,1],
                    [1,0,1],
                    [1,1,1]])
y = np.array([[0,0,1,1]]).T

#plt.matshow(np.hstack((X,y)), fignum= 10 ,cmap =plt.cm.gray)
#plt.show()

def nonlin(x, deriv=False):
    if deriv == True:
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))

Xaxis = np.arange(-5,5,0.2)

#plt.plot(Xaxis, nonlin(Xaxis))
#plt.show()
np.random.seed(1)
syn0 = 2*np.random.random((3,1))-1

for iter in range(15000):

    #feed forward
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))

    #error calc
    l1_error = y-l1
   
    l1_delta = l1_error * nonlin(l1, True)

    syn0+= np.dot(l0.T, l1_delta)

print("Output  :")
print(l1)




    
