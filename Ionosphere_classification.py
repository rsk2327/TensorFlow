import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

train = pd.read_csv("Ionosphere.csv")

label = train[['label']]
data = train[[col for col in train.columns if col not in ('label')]]

trainX,testX,trainY,testY = train_test_split(data,label,test_size=0.1,random_state=42)

trainData = np.array(trainX)
trainLabel = np.array(trainY)
testData = np.array(testX)
testLabel = np.array(testY)

from keras.utils import np_utils
trainLabel = np_utils.to_categorical(trainLabel)
testLabel = np_utils.to_categorical(testLabel)


import keras

from keras.models import Sequential
from keras.layers import Dense,Dropout

model = Sequential()
model.add(Dense(64, activation = "sigmoid", input_dim = 34))
model.add(Dropout(0.3))
model.add(Dense(32, activation = "sigmoid"))
model.add(Dense(2, activation = "softmax"))

model.compile(optimizer = 'rmsprop',
             loss = 'binary_crossentropy',
             metrics = ['accuracy'])

model.fit(trainData,trainLabel,epochs = 25, batch_size = 30)

print model.summary()

print model.get_config()

weights = model.get_weights()
print weights

testPred = model.predict(testData,verbose=1)

eval = model.evaluate(testData,testLabel,batch_size =1)

from keras import backend as K

modelGraph = K.get_session().graph
