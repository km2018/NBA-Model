import warnings

from astropy.modeling import optimizers
from keras.activations import *
from keras.initializers import *
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import *
from keras.wrappers.scikit_learn import *
from sklearn.ensemble import *
from sklearn.externals import joblib
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import scale
from sklearn.svm import *

import evaluateProfits as ep
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
data = pd.DataFrame.from_csv("2010-17data.csv").values
year = 2017

def getTrainData(year):
    l = len(data)
    if(year > 2014):
        X = data[l - 1230 * (2021-year):l - 1230 * (2018-year), 3:]
        Y = data[l - 1230 * (2021-year):l - 1230 * (2018-year), 0]
    else:
        X = data[l- 1230 * (2020-year)-990:l- 1230 * (2017-year), 3:]
        Y = data[l - 1230 * (2020-year)-990:l - 1230 * (2017-year), 0]

    return X,Y

def create_model(year=2017, optimizer = 'adam', activation = 'relu', neuron_config  = [64,64], 
                 epochs = 75, batch_size = 10, dropout_config = [0.7,0.7], 
                 weight_constraint = None, initializer = 'glorot_uniform'):
    from keras.constraints import max_norm
    # Load the datasets
    X,Y = getTrainData(year)
    test = pd.DataFrame.from_csv(str(year) + "data.csv").values
    pX, pY = test[:,3:22], test[:,0]

    # Create model
    model = Sequential()
    model.add(Dense(neuron_config[0], input_dim=19, kernel_initializer = initializer,
                    activation = activation, kernel_constraint = max_norm(weight_constraint)))
    
    model.add(Dropout(dropout_config[0]))
    model.add(Dense(neuron_config[1], activation = activation))
    model.add(Dropout(dropout_config[1]))
    
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, 
                  metrics=['accuracy'])
    
    # Fit the model
    model.fit(X, Y, epochs= epochs, batch_size=batch_size, verbose=2)
    
    scores = model.evaluate(pX, pY)
    print("\n%s: %.2f%%\n" % (model.metrics_names[1], scores[1] * 100))
    
    return model

def getmodelResults(year):
    bettingData = pd.DataFrame.from_csv(str(year) + 'data.csv')
    model = create_model(year)
    ep.getResults(model, year, bettingData)


#import evaluateProfits as ep
# master = pd.DataFrame.from_csv("2010-17data.csv")
# test = pd.DataFrame.from_csv('2013data.csv')
# X_test = test.iloc[:,3:22]
# y_test = test.iloc[:,0]
# X_train = master.iloc[:3450,3:22]
# y_train = master.iloc[:3450,0]                              
# 
# model = MLPClassifier(hidden_layer_sizes=(128,32,4), random_state =0)
#model = BaggingClassifier(MLPClassifier(hidden_layer_sizes=(128,32,4),
#                        random_state =0), max_samples=0.1, max_features=0.2)
# model = BaggingClassifier(KNeighborsClassifier(n_neighbors=100),
#                              max_samples=0.5, max_features=0.5)
# model = RandomForestClassifier(n_estimators=500)
# model = AdaBoostClassifier(n_estimators=100)
# model.fit(X_train, y_train)
# acc = model.score(X_test,y_test)
# print(acc)
#print(cross_val_score(model,X_train,y_train).mean())
# joblib.dump(mlp,'2013m.pkl')
# mlp = joblib.load('2013m.pkl')
#ep.getResults(mlp,test)

model = KerasClassifier(build_fn=create_model)
X,Y = getTrainData(year)
 
# activations = ['softmax','elu','selu','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid',
#                'linear']
dropout_configs = np.array([[[a/10, b/10] for b in range(10)] for a in range(10)])
dropout_configs.resize(1,100)
print(dropout_configs)
weight_constraints = np.arange(6)
initializers = ['RandomNormal','RandomUniform',"TruncatedNormal",'VarianceScaling',
                'Orthogonal','lecun_uniform','glorot_normal','glorot_uniform','he_normal','lecun_normal',
                'he_uniform']
param_grid = dict(dropout_config = dropout_configs, weight_constraint = weight_constraints)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X,Y)
 
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))