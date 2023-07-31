import numpy as np
import matplotlib.pyplot as plt #for plotting
from sklearn.datasets import make_regression
from sklearn import datasets
from perceptron import Perceptron
from sklearn.model_selection import train_test_split

losses=[]

X, Y, coef = datasets.make_regression(n_samples=100,n_features=1,n_informative=1,noise=10,coef=True,random_state=0)

X = np.interp(X, (X.min(), X.max()), (0, 20))

Y = np.interp(Y, (Y.min(), Y.max()), (20000, 150000))

X_train , X_test , Y_train , Y_test=train_test_split(X,Y, shuffle=True, test_size=0.2)

X_train = X_train.reshape(-1,1)
Y_train = Y_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)
Y_test = Y_test.reshape(-1,1)

# plt.ion()
# plt.plot(X,Y,'.',label='training data')
# plt.xlabel('Years of experience')
# plt.ylabel('Salary $')
# plt.title('Experience Vs. Salary')
# # plt.show()
# plt.pause(5)




perceptron= Perceptron(learning_rate_w=0.0001,learning_rate_b=0.1)
perceptron.fit(X_train,Y_train)
perceptron.evaluate(X_test,Y_test)
Y_pred=perceptron.predict(X_test)
Y_pred=Y_pred.reshape(X_train.shape[0],-1)