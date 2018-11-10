import warnings

from keras import optimizers
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.models import load_model
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def getProfit(moneyline):
    if(moneyline > 0):
        return moneyline
    else:
        return abs(10000 / moneyline)


start = 0
end = 3449
year = "2013"

unwanted_columns = ['Unnamed: 0', 'B_W/L', 'C_hostID', 'D_visitingID']
data = pd.read_csv("2010-17data.csv")
training = data[:][start:end].drop(unwanted_columns, axis = 1)
labels = data['B_W/L'][start:end]

# create model
model = Sequential()
model.add(Dense(64, input_dim=19, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

# Fit the model
model.fit(training, labels, epochs=125, batch_size=10, verbose=2)
model.save(year + "model.h5")
