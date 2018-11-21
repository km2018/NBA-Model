from keras import optimizers
from keras.layers import Dense, Dropout
from keras.models import Sequential

import pandas as pd

# Constants
COLUMN_AXIS = 1
NUM_FEATURES = -1

START_TRANING_DATA_ROW = 0
END_TRAINING_DATA_ROW = 3449
YEAR = "2013"

LABEL_COLUMN = 'B_W/L'
DATA_FILE_NAME = "2010-17data.csv"
AUTO_GENERATED_COLUMN = ['Unnamed: 0']
NON_NUMERIC_COLUMNS = ['B_W/L', 'C_hostID', 'D_visitingID']

NUM_NEURONS = 64
DROPOUT_SIZE = 0.7
OUTPUT_LAYER_SIZE = 1
INTERMEDIATE_ACTIVATION_FUNCTION = 'relu'
OUTPUT_ACTIVATION_FUNCTION = 'sigmoid'

LOSS = 'binary_crossentropy'
OPTIMIZER = 'adam'
METRICS = ['accuracy']

NUM_EPOCHS = 125
BATCH_SIZE = 10
VERBOSE = 2

MODEL_FILE_NAME = YEAR + "model.h5"

# Read in data into Pandas dataframes
data = pd.read_csv(DATA_FILE_NAME).drop(AUTO_GENERATED_COLUMN, axis = COLUMN_AXIS)
training = data[:][START_TRANING_DATA_ROW : END_TRAINING_DATA_ROW].drop(NON_NUMERIC_COLUMNS, axis = COLUMN_AXIS)
labels = data[LABEL_COLUMN][START_TRANING_DATA_ROW : END_TRAINING_DATA_ROW]

# Initialize number of features fed into neural network
NUM_FEATURES = len(training.columns)

# Add Layers to model
model = Sequential()
model.add(Dense(NUM_NEURONS, input_dim = NUM_FEATURES, activation = INTERMEDIATE_ACTIVATION_FUNCTION))
model.add(Dropout(DROPOUT_SIZE))
model.add(Dense(NUM_NEURONS, activation = INTERMEDIATE_ACTIVATION_FUNCTION))
model.add(Dropout(DROPOUT_SIZE))
model.add(Dense(OUTPUT_LAYER_SIZE, activation = OUTPUT_ACTIVATION_FUNCTION))

# Compile model
model.compile(loss = LOSS, optimizer = OPTIMIZER, metrics = METRICS)

# Fit the model
model.fit(training, labels, epochs = NUM_EPOCHS, batch_size = BATCH_SIZE, verbose = VERBOSE)
model.save(MODEL_FILE_NAME)
