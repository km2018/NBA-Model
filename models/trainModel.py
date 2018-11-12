from keras import optimizers
from keras.layers import Dense, Dropout
from keras.models import Sequential

import pandas as pd

# Constants
column_axis = 1
num_features = -1

start_traning_data_row = 0
end_training_data_row = 3449
year = "2013"

label_column = 'B_W/L'
data_file_name = "2010-17data.csv"
auto_generated_column = ['Unnamed: 0']
non_numeric_columns = ['B_W/L', 'C_hostID', 'D_visitingID']

num_neurons = 64
dropout_size = 0.7
output_layer_size = 1
intermediate_activation_function = 'relu'
output_activation_function = 'sigmoid'

loss = 'binary_crossentropy'
optimizer = 'adam'
metrics = ['accuracy']

num_epochs = 125
batch_size = 10
verbose = 2

model_file_name = year + "model.h5"

# Read in data into Pandas dataframes
data = pd.read_csv(data_file_name).drop(auto_generated_column, axis = column_axis)
training = data[:][start_traning_data_row : end_training_data_row].drop(non_numeric_columns, axis = column_axis)
labels = data[label_column][start_traning_data_row : end_training_data_row]

# Initialize number of features fed into neural network
num_features = len(training.columns)

# Add Layers to model
model = Sequential()
model.add(Dense(num_neurons, input_dim = num_features, activation = intermediate_activation_function))
model.add(Dropout(dropout_size))
model.add(Dense(num_neurons, activation = intermediate_activation_function))
model.add(Dropout(dropout_size))
model.add(Dense(output_layer_size, activation = output_activation_function))

# Compile model
model.compile(loss = loss, optimizer = optimizer, metrics = metrics)

# Fit the model
model.fit(training, labels, epochs = num_epochs, batch_size = batch_size, verbose = verbose)
model.save(model_file_name)
