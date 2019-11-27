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

start = timeit.default_timer()
print('Training Samples with Neural Network')

#load data from the matlab file
DataSets = sio.loadmat('DataSets.mat')
Xtrain = DataSets['Xtrain']
Xtest = DataSets['Xtest']

x_train = Xtrain[0,:]
t_train = Xtrain[1,:]
x_test = Xtest[0,:]
t_test = Xtest[1,:]
logsof_activation = ['sigmoid','softplus']

accuracy = np.zeros((10,10))
tenf = KFold(n_splits=10)
k = 0
for choose in logsof_activation:
    j = 0
    for train_indice, test_indice in tenf.split(x_train):
        
        x_sep_train, x_validation = x_train[train_indice], x_train[test_indice]
        t_sep_train, t_validation = t_train[train_indice], t_train[test_indice]
        for i in range(1,11):
            model = Sequential()
            
            model.add(Dense(i, activation=choose, input_dim=1))
            model.add(Dense(1, activation=None))
            sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='mean_squared_error', optimizer='adam')
                       
            converged = 0
            temp = 0
            epsilon = 0.01
            while not converged:
                model.fit(x_sep_train, t_sep_train, batch_size=128, epochs=100, verbose=0)
                score = model.evaluate(x_validation, t_validation, verbose=0)
                converged = np.abs(score-temp)<epsilon
                temp = score
                
            print("Score in Fold seperation detemination " + str(i))
            print(score)
            accuracy[j,i-1] = score
        print("Score in each train_validation determination " + str(score))
        j+=1
    
    if k == 0:
        logistic_accuracy = accuracy
    else:
        softplus_accuracy = accuracy
    k+=1
    accuracy = np.zeros((10,10))

model_means_log = np.mean(logistic_accuracy,axis=1)
model_order_log = np.argmin(model_means_log)+1

model_means_sof = np.mean(softplus_accuracy,axis=1)
model_order_sof = np.argmin(model_means_sof)+1



#Choose the better activation function between two traning model
if model_means_sof[model_order_sof-1] < model_means_log[model_order_log-1]:
    choose = 1
    model_preference = model_order_sof
else:
    choose = 0
    model_preference = model_order_log

model = Sequential()
model.add(Dense(model_preference, activation=logsof_activation[choose], input_dim=1))
model.add(Dense(1, activation=None))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error',optimizer='adam')
               
converged = 0
temp = 0
epsilon = 0.01
while not converged:
    model.fit(x_train, t_train, batch_size=128, epochs=100, verbose=0)
    score = model.evaluate(x_test, t_test, verbose=0)
    converged = np.abs(score-temp)<epsilon
    temp = score    
print("MSE on Test Data: " + str(score))
t_prediction = model.predict(x_test,batch_size=None)

model = Sequential()
model.add(Dense(4, activation='relu', input_dim=1))
model.add(Dense(1, activation=None))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer='adam')
 
converged = 0
temp = 0
epsilon = 0.01
while not converged:
    model.fit(x_train, t_train, batch_size=128, epochs=100, verbose=0)
    score = model.evaluate(x_test, t_test)
    converged = np.abs(score-temp)<epsilon
    temp = score    
print("MSE on Test Data in Smooth ReLu: " + str(score))
t_prediction_Relu = model.predict(x_test,batch_size=None)

stop = timeit.default_timer()
print('Running Time = ', stop-start)


plt.plot(np.arange(1,11), model_means_log,'r')
plt.plot(np.arange(1,11), model_means_sof,'g')
plt.title('1000 samples with 10-fold cross validation training in 1-10 Perceptron')
plt.xlabel('Number of Perceptrons')
plt.ylabel('Mean Square Error')
plt.legend(['Logistic', 'Softplus'])
plt.show()


plt.subplot(121)
plt.plot(x_test,t_test,'.')
plt.title('Original Data')
plt.xlabel('x1'); plt.ylabel('Target = x2')
plt.subplot(122)
plt.title('Neural Network Output')
plt.plot(x_test, t_prediction, '.')
plt.xlabel('x1');
plt.show()


plt.subplot(121)
plt.plot(x_test,t_test,'.')
plt.title('Original Data')
plt.xlabel('x1'); plt.ylabel('Target = x2')
plt.subplot(122)
plt.title('Neural Network Output with ReLu')
plt.plot(x_test, t_prediction, '.')
plt.xlabel('x1');
plt.show()






