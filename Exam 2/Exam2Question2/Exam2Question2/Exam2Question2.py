import time
import timeit
import tensorflow as tf
import numpy as np
import keras
import scipy.io as sio
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

print('Training Samples with Neural Network')

#load data from the matlab file
DataSets = sio.loadmat('DataSets.mat')
Xtrain = DataSets['Xtrain']
Xtest = DataSets['Xtest']

x_train = Xtrain[0,:]
t_train = Xtrain[1,:]
x_test = Xtest[0,:]
t_test = Xtest[1,:]

accuracy = np.zeros((10,10))
tenf = KFold(n_splits=10)
j = 0
for train_indice, test_indice in tenf.split(x_train):
        
        x_sep_train, x_validation = x_train[train_indice], x_train[test_indice]
        t_sep_train, t_validation = t_train[train_indice], t_train[test_indice]
        for i in range(1,11):
            model = Sequential()
            
            model.add(Dense(i, activation='sigmoid', input_dim=1))
            model.add(Dense(1, activation=None))
            sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
            model.compile(loss='mean_squared_error', optimizer='adam')
                       
            converged = 0
            temp = 0
            epsilon = 0.01
            while not converged:
                model.fit(x_sep_train, t_sep_train, batch_size=128, epochs=100, verbose=0)
                score = model.evaluate(x_validation, t_validation, verbose=0)
                converged = np.abs(score-temp)<epsilon
                temp = score
            print(score)
            accuracy[j,i-1] = score
        j+=1
logistic_accuracy = accuracy
model_means_log = np.mean(accuracy,axis=1)
model_order_log = (np.argmin(model_means)+1)

accuracy = np.zeros((10,10))
j = 0
for train_indice, test_indice in tenf.split(x_train):
        
        x_sep_train, x_validation = x_train[train_indice], x_train[test_indice]
        t_sep_train, t_validation = t_train[train_indice], t_train[test_indice]
        for i in range(1,11):
            model = Sequential()
            
            model.add(Dense(i, activation='softplus', input_dim=1))
            model.add(Dense(1, activation=None))
            sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
            model.compile(loss='mean_squared_error', optimizer='adam')
                       
            converged = 0
            temp = 0
            epsilon = 0.01
            while not converged:
                model.fit(x_sep_train, t_sep_train, batch_size=128, epochs=100, verbose=0)
                score = model.evaluate(x_validation, t_validation, verbose=0)
                converged = np.abs(score-temp)<epsilon
                temp = score
            print(score)
            accuracy[j,i-1] = score
        j+=1
softplus_accuracy = accuracy
model_means_soft = np.mean(accuracy,axis=1)
model_order_sof = (np.argmin(model_means)+1)

#Apply to testing of logistic
model = Sequential()          
model.add(Dense(model_order_log, activation='sigmoid', input_dim=1))
model.add(Dense(1, activation=None))
sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer='adam')
                       
converged = 0
temp = 0
epsilon = 0.01
while not converged:
    model.fit(x_train, t_train, batch_size=128, epochs=100, verbose=0)
    score = model.evaluate(x_test, t_test, verbose=0)
    converged = np.abs(score-temp)<epsilon
    temp = score
print(score)

print("MSE on Test Data with Logistic: " + str(score))

#Apply to testing of softplus
model = Sequential()          
model.add(Dense(model_order_sof, activation='softplus', input_dim=1))
model.add(Dense(1, activation=None))
sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer='adam')
                       
converged = 0
temp = 0
epsilon = 0.01
while not converged:
    model.fit(x_train, t_train, batch_size=128, epochs=100, verbose=0)
    score = model.evaluate(x_test, t_test, verbose=0)
    converged = np.abs(score-temp)<epsilon
    temp = score
print(score)

print("MSE on Test Data with Softplus: " + str(score))

plt.plot(np.arange(1,11), model_means_log,'r')
plt.plot(np.arange(1,11), model_means_sof,'g')
plt.title('1000 samples with 10-fold cross validation training in 1-10 Perceptron')
plt.xlabel('Number of Perceptrons')
plt.ylabel('Mean Square Error')
plt.legend(['Sigmoid', 'Softplus'])
plt.show()

