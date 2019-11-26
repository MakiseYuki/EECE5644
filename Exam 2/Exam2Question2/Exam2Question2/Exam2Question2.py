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
logsof_activation = ['sigmoid','softplus']

accuracy = np.zeros((10,10))
tenf = KFold(n_splits=10)
k=0
for choose in logsof_activation:
    j = 0
    for train_indice, test_indice in tenf.split(x_train):
        
            x_sep_train, x_validation = x_train[train_indice], x_train[test_indice]
            t_sep_train, t_validation = t_train[train_indice], t_train[test_indice]
            for i in range(1,11):
                model = Sequential()
            
                model.add(Dense(i, activation=choose, input_dim=1))
                model.add(Dense(1, activation=None))
                sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
                model.compile(loss='mean_squared_error', optimizer='adam')
                       
                converged = 0
                temp = 0
                epsilon = 0.11
                while not converged:
                    model.fit(x_sep_train, t_sep_train, batch_size=128, epochs=20, verbose=0)
                    score = model.evaluate(x_validation, t_validation, verbose=0)
                    converged = np.abs(score-temp)<epsilon
                    temp = score
                
                print("Score in Fold seperation" + str(i))
                print(score)
                accuracy[j,i-1] = score
            j+=1
    if i == 0:
        logistic_accuracy = accuracy
    else:
        softplus_accuracy = accuracy
k+=1

model_means_log = np.mean(logistic_accuracyy,axis=1)
model_means_sof = np.mean(softplus_accuracy, axis=1)
model_order_log = (np.argmin(model_means_log)+1)
model_order_sof = (np.argmin(model_means_sof)+1)

#Choose the better activation function between two traning model
if model_means_sof[model_order_soft-1] < model_means_log[model_order_log-1]:
    choose = 1
    model_order = model_order_soft
else:
    choose = 0
    model_order = model_order_log

model = Sequential()
model.add(Dense(model_order, activation=logsof_activation[choose], input_dim=1))
model.add(Dense(1, activation=None))
sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error',optimizer='adam')
               
converged = 0
tmp = 0
epsilon = 0.11
while not converged:
    model.fit(x_train, t_train, batch_size=128, epochs=20)
    score = model.evaluate(x_test, t_test)
    print(score)
    print(temp)
    converged = np.abs(score-temp)<epsilon
    temp = score    
print("MSE on Test Data: " + str(score))
t_prediction = model.predict(x_test,batch_size=None)


plt.plot(np.arange(1,11), model_means_log,'r')
plt.plot(np.arange(1,11), model_means_sof,'g')
plt.title('1000 samples with 10-fold cross validation training in 1-10 Perceptron')
plt.xlabel('Number of Perceptrons')
plt.ylabel('Mean Square Error')
plt.legend(['Sigmoid', 'Softplus'])
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







