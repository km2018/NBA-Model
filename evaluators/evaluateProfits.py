import warnings

from keras import optimizers
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.models import load_model
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def getProfit(moneyline):
    if(moneyline > 0):
        return moneyline
    else:
        return abs(10000/moneyline)

year = '2017'
segment = ""
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
data = pd.read_csv(year + "data.csv").drop(["Unnamed: 0"], axis = 1)
#predict = np.loadtxt("2017-18data.csv", dtype = str, delimiter = ",")

#IDs of the two teams you are predicting
# host_ID = '1610612759'
# road_ID = '1610612753'
#  

# X = data[1:, 4:]
# Y = data[1:, 1]

# pX = predict[1:, 4:]
# pY = predict[1:, 1]

# create model
#model = Sequential()
model = load_model(year + "model.h5")

maxReturns = 0
EVs = []
lowerBounds = []
upperBounds = []
tProfits = []
percentBetted = []
percentCorrect = []
lowestGame = []
returns = []
lowestMarginalProfit = []
for x in range(0, 5):
    EVFloor = x * 4                                     # CHANGED THIS LINE
    for m in range (4):
        totalProfit = 0
        gamesBetted = 0
        predLower = 0.05 + ((m*10)/100)
        predUpper = 0.95 - ((m*10)/100)
        correct = 0
        incorrect = 0 
        lowProf = 200000
        lowGame = -1

        for i in range (0, int(len(data))):
            inputList = np.array(data.iloc[i,3:22])
            inputList.resize((1,19))
            
            mPred = model.predict(inputList)[0][0]
            pred = round(mPred)
            profit = -100

            gameResult = data.iloc[i,0]
            temp_homeline = data.iloc[i,22]
            temp_roadline = data.iloc[i,23]
            if(temp_homeline == "NL" or temp_roadline == "NL"):
                continue
            homeline = int(temp_homeline)
            roadline = int(temp_roadline)

            homeEV = getProfit(homeline) * mPred - (100 * (1 - mPred))
            roadEV = (getProfit(roadline) * (1 - mPred)) - (100 * mPred)
            
            if(homeEV < EVFloor and roadEV < EVFloor):
                continue
            pred = int(homeEV > roadEV)
            
            if(mPred > predLower and mPred < predUpper):   
                if(pred and gameResult):
                    profit = getProfit(homeline)
#                   print(str(profit) + "   " + str(totalProfit))
#                   time.sleep(2)
                    correct += 1
                elif(not pred and not gameResult):
                    profit = getProfit(roadline)
#                   print(str(profit) + "   " + str(totalProfit))
#                   time.sleep(2)
                    correct += 1
                else:
                    profit = -100
#                   print(str(profit) + "   " + str(totalProfit))
#                   time.sleep(2)
                    incorrect += 1
                gamesBetted += 1
            else:
                profit = 0
            
            totalProfit += profit
            if(totalProfit < lowProf):
                lowProf = totalProfit
                lowGame = i + 1


#       time.sleep(300)
        print("Number Correct: %d" % (correct))
        print("Number Incorrect: %d" % (incorrect))

        if((totalProfit/gamesBetted) > maxReturns):
            maxReturns = (totalProfit/gamesBetted)

        tProfits.append(totalProfit)
        EVs.append(EVFloor)
        returns.append(totalProfit/gamesBetted)
        percentBetted.append(gamesBetted/1230)
        lowerBounds.append((predLower) * 100)
        upperBounds.append((predUpper) * 100)
        lowestMarginalProfit.append(lowProf)
        lowestGame.append(lowGame)
        percentCorrect.append(correct/gamesBetted)

        print("Expected value floor: %d\nLower bound: %f, Upper bound: %f \nTotal Profit for 2016 season is: %.2f \nAverage profit is: %.2f" 
              % (EVFloor,predLower,predUpper,totalProfit,totalProfit/gamesBetted))
        print("Percentage of Games Betted: %.2f" % (gamesBetted/1229)) 
    print("Maximum Returns: %f" % (maxReturns))

results = pd.DataFrame(data = EVs, columns = ['EV'])
results['Lower Bound'] = lowerBounds
results['Upper Bound'] = upperBounds
results['Return'] = returns
results['Profits'] = tProfits
results['Percent Betted'] = percentBetted
results['Percent Correct'] = percentCorrect
results['Biggest Hole'] = lowestMarginalProfit
results['Lowest Profit Game Number'] = lowestGame
results.to_csv(year + "results" + segment + ".csv")
# model.add(Dense(64, input_dim = 14, activation='relu'))
# model.add(Dropout(0.7))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.7))
# model.add(Dense(1, activation='sigmoid'))
#      
# # Compile model
# model.compile(loss='binary_crossentropy', optimizer= "adam", metrics=['accuracy'])
#      
# # Fit the model
# model.fit(X, Y, epochs= 100, batch_size= 10, verbose=2)
# model.save("2017model.h5")

# scores = model.evaluate(X, Y)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
# 
# # calculate predictions
# scores = model.evaluate(pX, pY)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# df = pd.DataFrame(model.get_weights())
# df.to_csv('my_model_weights.csv')
# 
# for r in library:
#     if (r[0] == host_ID):
#         host = r[1:]
#     if (r[0] == road_ID):
#         road = r[1:]
# x1 = np.concatenate((road,host)) 
# x1 = x1.reshape(1, 16)
# print (model.predict(x1))

