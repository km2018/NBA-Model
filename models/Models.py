import warnings

from astropy.modeling import optimizers
from keras.activations import *
from keras.initializers import *
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import *
from keras.wrappers.scikit_learn import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category = np.VisibleDeprecationWarning)

DATE_FILE_NAME = "2010-17data.csv"

DATA = pd.DataFrame.from_csv(DATA_FILE_NAME).values
YEAR = 2017
CURR_YEAR = 2017
OUTLIER_YEAR = 2011
GAMES_REGULAR = 1230
GAMES_2011 = 990
YEARS_OF_TRAIN_DATA = 3

START_COLUMN = 3
END_COLUMN = 22
LABEL_COLUMN = 0

'''
    Parameters:
        year: season of interest
    Returns:
        two arrays, X and Y, representing training features and labels
'''
def getTrainData(YEAR):

    total_games = len(DATA)

    if(YEAR > OUTLIER_YEAR + YEARS_OF_TRAIN_DATA):
        X = DATA[total_games - GAMES_REGULAR * (CURR_YEAR + YEARS_OF_TRAIN_DATA - YEAR + 1):
                 total_games - GAMES_REGULAR * (CURR_YEAR - YEAR + 1), START_COLUMN : END_COLUMN]

        Y = DATA[total_games - GAMES_REGULAR * (CURR_YEAR + YEARS_OF_TRAIN_DATA - YEAR + 1):
                 total_games - GAMES_REGULAR * (CURR_YEAR - YEAR + 1), LABEL_COLUMN]

    else:
        X = DATA[total_games - GAMES_REGULAR * (CURR_YEAR + YEARS_OF_TRAIN_DATA - YEAR) - GAMES_2011:
                 total_games - GAMES_REGULAR * (CURR_YEAR - YEAR), START_COLUMN : END_COLUMN]

        Y = DATA[total_games - GAMES_REGULAR * (CURR_YEAR - YEAR) - GAMES_2011:
                 total_games - GAMES_REGULAR * (CURR_YEAR - YEAR), LABEL_COLUMN]

    return X, Y

def train_and_eval_model(year = 2017, optimizer = 'adam', activation = 'relu', neuron_config = [64,64],
                         epochs = 75, batch_size = 10, dropout_config = [0.7,0.7],
                         weight_constraint = None, initializer = 'glorot_uniform', verbose = 2):
    '''
        Parameters:
            year: season of interest
            optimizer: optimizer of choice
            activation: layer activation of choice
            neuron_config: an array representing # of neurons in each layer
            epochs: # of epochs to train
            batch_size: batch size
            dropout_config: an array representing percentage of dropout in each dropout layer
            weight_constraint: the weight for neurons
            initializer: initializer of choice that initializes the weights
            verbose: verbose level
        Returns:
            A trained keras model
    '''
    from keras.constraints import max_norm

    # Load the datasets
    X,Y = getTrainData(YEAR)
    test = pd.DataFrame.from_csv(str(YEAR) + "data.csv").values
    pX, pY = test[:, START_COLUMN : END_COLUMN], test[:, LABEL_COLUMN]

    # Create model
    model = Sequential()
    model.add(Dense(neuron_config[0], input_dim = len(X[0]), kernel_initializer = initializer,
                    activation = activation, kernel_constraint = max_norm(weight_constraint)))
    model.add(Dropout(dropout_config[0]))
    model.add(Dense(neuron_config[1], activation = activation))
    model.add(Dropout(dropout_config[1]))

    model.add(Dense(1, activation = 'sigmoid'))

    # Compile model
    model.compile(loss = 'binary_crossentropy', optimizer = optimizer,
                  metrics = ['accuracy'])

    # Fit the model
    model.fit(X, Y, epochs = epochs, batch_size = batch_size, verbose = verbose)

    scores = model.evaluate(pX, pY)
    print("\n%s: %.2f%%\n" % (model.metrics_names[1], scores[1] * 100))

    return model

def gridSearch():
    #Grid Search Codes to find best config#

    ###########################TO BE UPDATED###################################

    model = KerasClassifier(build_fn=create_model)
    X,Y = getTrainData(year)

    # activations = ['softmax','elu','selu','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid',
    #                'linear']
    dropout_configs = np.array([[[a / 10, b / 10] for b in range(10)] for a in range(10)])
    dropout_configs.resize(1, 100)
    print(dropout_configs)
    weight_constraints = np.arange(6)
    initializers = ['RandomNormal','RandomUniform',"TruncatedNormal",'VarianceScaling',
                    'Orthogonal','lecun_uniform','glorot_normal','glorot_uniform','he_normal','lecun_normal',
                    'he_uniform']

    param_grid = dict(dropout_config = dropout_configs, weight_constraint = weight_constraints)
    grid = GridSearchCV(estimator = model, param_grid = param_grid)
    grid_result = grid.fit(X,Y)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    ###########################TO BE UPDATED###################################

#Main function that train and evaluate and save the model
def main():
    model = train_and_eval_model(year = YEAR)
    model.save(str(YEAR) + ".h5")

#Allows for directly running main() from terminal with "python Models.py"
if __name__ == '__main__':
    main()
