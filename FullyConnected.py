import numpy as np
import tensorflow as tf

class Dense:
    def __init__(self, inSize, outSize, lr):
        std = .1
        self._w = np.random.normal(loc=0.0, scale=std, size=(inSize, outSize))
        self._b = np.zeros((outSize))
        self._lr = lr

    def Forward(self, x):
        #storing input for backward pass
        self._x = x
        z = x @ self._w + self._b
        self._y = self._Sigmoid(z)
        return self._y

    def Backward(self, dy):
        #intermediate chain rule calculation
        dz = dy * self._y * (1 - self._y)
        #parameter gradient calculations
        db = np.sum(dz, axis=0) 
        dw = self._x.T @ dz
        #input gradient calculation used to pass the chain rule back further in the network
        dx = dz @ self._w.T
        #parameter updates
        self._w -= (dw * self._lr)
        self._b -= (db * self._lr)
        return dx

    def _Sigmoid(self, z):
        return 1/(1 + np.exp(-z))

class LossLayer:
    def __init__(self):
        pass

    def Forward(self, y, yhat):
        #store y and yhat for the backward pass
        self._y = y
        self._yhat = yhat
        #batch loss calculations
        squaredErr = (y - yhat) ** 2
        loss = np.sum(squaredErr)
        #number correct in each batch
        temp1 = np.argmax(y, axis=-1)
        temp2 = np.argmax(yhat, axis=-1)
        boolArr = np.equal(temp1, temp2)
        intArr = boolArr * 1
        numCorrect = np.sum(intArr)
        return loss, numCorrect

    def Backward(self):
        dyhat = -2/np.size(self._yhat) * (self._y - self._yhat)
        return dyhat

########MAIN PROGRAM###################

#DATA PREPERATION
# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
# Split the data into training and testing sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize the pixel values to be between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0
#cast the data sets into the correct types
x_train = tf.cast(x_train, dtype=tf.float16)
x_test = tf.cast(x_test, dtype=tf.float16)
#reshape the data
x_train = tf.reshape(x_train,(x_train.shape[0], -1)).numpy()
x_test = tf.reshape(x_test,(x_test.shape[0], -1)).numpy()
#turning the labels into one hot enc
num_classes = 10
y_train_one_hot = tf.one_hot(y_train, depth=num_classes).numpy()
y_test_one_hot = tf.one_hot(y_test, depth=num_classes).numpy()

#NETWORK SETUP
dense1 = Dense(inSize=784, outSize=16, lr=1.0)
dense2 = Dense(inSize=16, outSize=10, lr=1.0)
lossLayer = LossLayer()

#HYPER PARAMETER INIT
numTrainExamples = x_train.shape[0]
numTestExamples = x_test.shape[0]
numEpochs = 100
batchSize = 100
numBatches = int(numTrainExamples / batchSize)

for epoch in range(numEpochs):
    totalLoss = 0
    totalCorrect = 0
    for batch in range(numBatches):
        #extract the relevant data for the current batch
        startIndex = batch * batchSize
        x_batch = x_train[startIndex : startIndex + batchSize]
        y_batch = y_train_one_hot[startIndex : startIndex + batchSize]
        #Forward pass
        y1 = dense1.Forward(x_batch)
        yhat = dense2.Forward(y1)
        loss, numCorrect = lossLayer.Forward(y=y_batch, yhat=yhat)
        totalLoss += loss
        totalCorrect += numCorrect
        #Backward pass
        dyhat = lossLayer.Backward()
        dy1 = dense2.Backward(dyhat)
        dense1.Backward(dy1)
        print("\rEpoch progress: ", round((batch / numBatches) * 100, 2), "%", end="")
    #Display the results from the epoch
    print(" Epoch: ", epoch + 1, " Loss: ", round(totalLoss / x_train.shape[0], 4), " Proportion Correct: ", round(totalCorrect/numTrainExamples,2))

#Testing
y1 = dense1.Forward(x_test)
yhat = dense2.Forward(y1)
loss, numCorrect = lossLayer.Forward(y=y_test_one_hot, yhat=yhat)
print("Test Complete! Correct Percentage: ", round(numCorrect/numTestExamples * 100, 2))
print("Execution Finished")