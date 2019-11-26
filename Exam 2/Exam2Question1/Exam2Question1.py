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
#from tensorflow.python.client import device_lib

#config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
#sess = tf.Session(config=config) 
#keras.backend.set_session(sess)
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#print(device_lib.list_local_devices())
print('Training Samples with Neural Network')
start = timeit.default_timer()

TEST = 100

if TEST == 100:
    #print('100 training case')
    trainDataLD_100 = sio.loadmat('trainData_100.mat')
    trainData_100 = trainDataLD_100['trainData_100']
    trainLabel_100 = trainDataLD_100['trainLabel_100']
   
    x_data = trainData_100.transpose()
    trainLabel_100 = trainLabel_100-1; #indice in matlab  1 is 0 in python
    trainLabel_100 = trainLabel_100[0] #indice in matlab  1 is 0 in python
    y_data = keras.utils.to_categorical(trainLabel_100, num_classes=4) 
    accuracy = np.zeros((10,10))
    #print(y_data[10])

    tenf = KFold(n_splits=10)
    j = 0

    print("Training for 100 samples...")
    for train_indice, test_indice in tenf.split(x_data):
        #print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = x_data[train_indice], x_data[test_indice]
        y_train, y_test = y_data[train_indice], y_data[test_indice]
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
                model.fit(x_train, y_train, batch_size=None, epochs=50, verbose=0)
                score = model.evaluate(x_test, y_test, verbose=0)
                converged = np.abs(score[1]-tmp)<epsilon
                tmp = score[1]
            accuracy[j,i-1] = score[1]
        j+=1

    print("Accuracy Rate on Train Data 100: "+ str(score[1]))
    #print(accuracy)

    model_means = np.mean(accuracy,axis=1)
    model_order = np.where(model_means == np.max(model_means))[0][0] + 1
    #model_order

    #plt.plot(np.arange(1,11), model_means, 'r')
    #plt.title('number of sample = 100, epochs = 100 with 10-fold cross validation traning')
    #plt.xlabel('perceptrons of each fold')
    #plt.ylabel('probability of correct rate')
    #plt.show()

    time.sleep(5)

    testDataLD_100 = sio.loadmat('testData_100.mat')
    testData_100 = testDataLD_100['testData_100']
    testLabel_100 = testDataLD_100['testLabel_100']
   
    
    x_test = testData_100.transpose()
    testLabel_100 = testLabel_100-1; #indice in matlab  1 is 0 in python
    testLabel_100 = testLabel_100[0] #indice in matlab  1 is 0 in python
    y_test = keras.utils.to_categorical(testLabel_100, num_classes=4) 
    model = Sequential()
    
    model.add(Dense(model_order, activation='tanh', input_dim=3))
    model.add(Dense(4, activation='softplus'))
    sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    converged = 0
    tmp = 0
    epsilon = 0.001
    while not converged:
        model.fit(x_data, y_data, batch_size=None, epochs=50, verbose=0)
        score = model.evaluate(x_test, y_test,verbose=0)
        converged = np.abs(score[1]-tmp)<epsilon
        tmp = score[1]

    prediction_method = model.predict(x_test)
    prediction = np.argmax(prediction_method,axis=1)
    con_matrix = confusion_matrix(testLabel_100, prediction, labels=[0, 1, 2, 3])
    print(score)
    print("Accuracy Rate on Test Data 100: " + str(score[1]))
    print("The confusion matrix is as below: ")
    print(con_matrix)

elif TEST == 1000:
  
    trainDataLD_1000 = sio.loadmat('trainData_1000.mat')
    trainData_1000 = trainDataLD_1000['trainData_1000']
    trainLabel_1000 = trainDataLD_1000['trainLabel_1000']
    
    x_data = trainData_1000.transpose()
    trainLabel_1000 = trainLabel_1000-1; #indice in matlab  1 is 0 in python
    trainLabel_1000 = trainLabel_1000[0] #indice in matlab  1 is 0 in python
    y_data = keras.utils.to_categorical(trainLabel_1000, num_classes=4) 
    accuracy = np.zeros((10,10))
    #print(y_data[10])

    tenf = KFold(n_splits=10)
    j = 0

    print("Training for 1000 samples...")
    for train_indice, test_indice in tenf.split(x_data):
        #print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = x_data[train_indice], x_data[test_indice]
        y_train, y_test = y_data[train_indice], y_data[test_indice]
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
                model.fit(x_train, y_train, batch_size=None, epochs=50, verbose=0)
                score = model.evaluate(x_test, y_test, verbose=0)
                converged = np.abs(score[1]-tmp)<epsilon
                tmp = score[1]
            accuracy[j,i-1] = score[1]
        j+=1

    print("Accuracy Rate on Train Data 1000: "+ str(score[1]))
    #print(accuracy)

    model_means = np.mean(accuracy,axis=1)
    model_order = np.where(model_means == np.max(model_means))[0][0] + 1
    #model_order

    plt.plot(np.arange(1,11), model_means, 'r')
    plt.title('Number of Sample = 1000, Epochs = 100 With 10-Fold Cross Validation Traning')
    plt.xlabel('Perceptrons of each Fold')
    plt.ylabel('Probability of Correct Rate')
    plt.show()

    time.sleep(5)

    testDataLD_1000 = sio.loadmat('testData_1000.mat')
    testData_1000 = testDataLD_1000['testData_1000']
    testLabel_1000 = testDataLD_1000['testLabel_1000']
   
    
    x_test = testData_1000.transpose()
    testLabel_1000 = testLabel_1000-1; #indice in matlab  1 is 0 in python
    testLabel_1000 = testLabel_1000[0] #indice in matlab  1 is 0 in python
    y_test = keras.utils.to_categorical(testLabel_1000, num_classes=4) 
    model = Sequential()
    
    model.add(Dense(model_order, activation='tanh', input_dim=3))
    model.add(Dense(4, activation='softplus'))
    sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    converged = 0
    tmp = 0
    epsilon = 0.001
    while not converged:
        model.fit(x_data, y_data, batch_size=None, epochs=50, verbose=0)
        score = model.evaluate(x_test, y_test,verbose=0)
        converged = np.abs(score[1]-tmp)<epsilon
        tmp = score[1]

    prediction_method = model.predict(x_test)
    prediction = np.argmax(prediction_method,axis=1)
    con_matrix = confusion_matrix(testLabel_1000, prediction, labels=[0, 1, 2, 3])
    print(score)
    print("Accuracy Rate on Test Data 1000: " + str(score[1]))
    print("The confusion matrix is as below: ")
    print(con_matrix)

stop = timeit.default_timer()
print('Running Time = ', stop-start)