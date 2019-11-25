import tensorflow as tf
import numpy as np
import keras
import scipy.io as sio
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.model_selection import KFold
from tensorflow.python.client import device_lib

#config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
#sess = tf.Session(config=config) 
#keras.backend.set_session(sess)
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#print(device_lib.list_local_devices())

trainDataLD_1000 = sio.loadmat('trainData_1000.mat')
trainDataLD_10000 = sio.loadmat('trainData_10000.mat')


trainData_1000 = trainDataLD_1000['trainData_1000']
trainData_10000 = trainDataLD_10000['trainData_10000']


trainLabel_1000 = trainDataLD_1000['trainLabel_1000']
trainLabel_10000 = trainDataLD_10000['trainLabel_10000']

TEST = 10000

if TEST==100:
    print('100 training case')
    trainDataLD_100 = sio.loadmat('trainData_100.mat')
    trainData_100 = trainDataLD_100['trainData_100']
    trainLabel_100 = trainDataLD_100['trainLabel_100']
    #100 sample size training
    
    x_data = trainData_100.transpose()
    trainLabel_100 = trainLabel_100-1; #indice in matlab  1 is 0 in python
    trainLabel_100 = trainLabel_100[0] #indice in matlab  1 is 0 in python
    y_data = keras.utils.to_categorical(trainLabel_100, num_classes=4) 
    accuracy = np.zeros((10,10))
    #print(y_data[10])

    kf = KFold(n_splits=10)
    j = 0
    for train_index, test_index in kf.split(x_data):
        #print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        for i in range(1,11):
            model = Sequential()
            
            model.add(Dense(i, activation='tanh', input_dim=3))
            model.add(Dense(4, activation='softplus'))
            sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
            model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
            converged = 0
            tmp = 0
            epsilon = 0.001
            while not converged:
                model.fit(x_train, y_train, batch_size=None, epochs=100, verbose=0)
                score = model.evaluate(x_test, y_test, verbose=0)
                converged = np.abs(score[1]-tmp)<epsilon
                tmp = score[1]
            accuracy[j,i-1] = score[1]
        j+=1

    print("Accuracy Rate on Train Data:"+ str(score[1]))
    #print(accuracy)

    model_means = np.mean(accuracy,axis=1)
    model_order = np.where(model_means == np.max(model_means))[0][0] + 1
    model_order

    plt.plot(np.arange(1,11), model_means, 'r')
    plt.title('Number of Sample = 100, Epochs = 100 With 10-Fold Cross Validation Traning')
    plt.xlabel('Perceptrons')
    plt.ylabel('Probability of Correct Rate')
    plt.show()

    y_testfinal = keras.utils.to_categorical(trainLabel_100, num_classes = 4)
    model = Sequential()
    
    model.add(Dense(model_order, activation='tanh', input_dim=3))
    model.add(Dense(4, activation='softplus'))
    sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    converged = 0
    tmp = 0
    epsilon = 0.001
    while not converged:
        model.fit(x_data, y_data, batch_size=None, epochs=100, verbose=0)
        score = model.evaluate(x_data, y_testfinal,verbose=0)
        converged = np.abs(score[1]-tmp)<epsilon
        tmp = score[1]
    print("Accuracy Rate on Test Data: " + str(score[1]))


elif TEST==10000:
    print('10000 training case')
    trainDataLD_10000 = sio.loadmat('trainData_10000.mat')
    trainData_10000 = trainDataLD_10000['trainData_10000']
    trainLabel_10000 = trainDataLD_10000['trainLabel_10000']
    #10000 sample size training
    
    x_data = trainData_10000.transpose()
    trainLabel_10000 = trainLabel_10000-1; #indice in matlab  1 is 0 in python
    trainLabel_10000 = trainLabel_10000[0] #indice in matlab  1 is 0 in python
    y_data = keras.utils.to_categorical(trainLabel_10000, num_classes=4) 
    accuracy = np.zeros((10,10))
    #print(y_data[10])

    kf = KFold(n_splits=10)
    j = 0
    for train_index, test_index in kf.split(x_data):
        #print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        for i in range(1,11):
            model = Sequential()
            
            model.add(Dense(i, activation='tanh', input_dim=3))
            model.add(Dense(4, activation='softplus'))
            sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
            model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
            converged = 0
            tmp = 0
            epsilon = 0.001
            while not converged:
                model.fit(x_train, y_train, batch_size=None, epochs=100, verbose=0)
                score = model.evaluate(x_test, y_test, verbose=0)
                converged = np.abs(score[1]-tmp)<epsilon
                tmp = score[1]
            accuracy[j,i-1] = score[1]
        j+=1

    print("Accuracy Rate on Train Data:"+ str(score[1]))
    #print(accuracy)

    model_means = np.mean(accuracy,axis=1)
    model_order = np.where(model_means == np.max(model_means))[0][0] + 1
    model_order

    plt.plot(np.arange(1,11), model_means, 'r')
    plt.title('Number of Sample = 10000, Epochs = 100 With 10-Fold Cross Validation Traning')
    plt.xlabel('Perceptrons')
    plt.ylabel('Probability of Correct Rate')
    plt.show()

    y_testfinal = keras.utils.to_categorical(trainLabel_10000, num_classes = 4)
    model = Sequential()
    
    model.add(Dense(model_order, activation='tanh', input_dim=3))
    model.add(Dense(4, activation='softplus'))
    sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    converged = 0
    tmp = 0
    epsilon = 0.001
    while not converged:
        model.fit(x_data, y_data, batch_size=None, epochs=100, verbose=0)
        score = model.evaluate(x_data, y_testfinal,verbose=0)
        converged = np.abs(score[1]-tmp)<epsilon
        tmp = score[1]
    print("Accuracy Rate on Test Data: " + str(score[1]))
