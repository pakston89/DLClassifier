import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

class genderClassifier():
    
    def __init__(self, model):
        self.model = model

    def ANNDefinition(self):
        #Defining ANN
        self.model = Sequential()
        self.model.add(Dense(4, input_shape=(8, 2), activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        opt = tf.keras.optimizers.SGD(learning_rate=0.1)
        self.model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

    def train(self, train_X, train_Y):
        #We train the model
        history = self.model.fit(train_X, train_Y, epochs=2500)

    def predict(self, test_X):
        y_pred = self.model.predict(test_X)
        print(y_pred)