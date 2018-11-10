import warnings

from keras import optimizers
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.models import load_model
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#calculate profit assuming bet is won
def getProfit(moneyline):
    if(moneyline > 0):
        return moneyline
    else:
        return abs(10000/moneyline)

#read data from csv for specified year
YEAR = '2017'
segment = ""
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
data = pd.read_csv(YEAR + "data.csv").drop(["Unnamed: 0"], axis = 1)

#load previously made model
model = load_model(YEAR + "model.h5")

#loop through year and calculate profit based on model's predictions
#declare and initialize arrays to hold betting data
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

#calculate profits for EV floor from 0-20 (increments of 4)
#for each EV floor calculate profits for prob windows from 0.05-0.95 to 0.45-0.55
for x in range(0, 5):
    EVFloor = x * 4                                     
    for m in range (4):
        #hold more data to be used later
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
                    correct += 1
                elif(not pred and not gameResult):
                    profit = getProfit(roadline)
                    correct += 1
                else:
                    profit = -100
                    incorrect += 1
                gamesBetted += 1
            else:
                profit = 0
            
            totalProfit += profit
            if(totalProfit < lowProf):
                lowProf = totalProfit
                lowGame = i + 1

        #print results data
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

#add results data to specified csv
results = pd.DataFrame(data = EVs, columns = ['EV'])
results['Lower Bound'] = lowerBounds
results['Upper Bound'] = upperBounds
results['Return'] = returns
results['Profits'] = tProfits
results['Percent Betted'] = percentBetted
results['Percent Correct'] = percentCorrect
results['Biggest Hole'] = lowestMarginalProfit
results['Lowest Profit Game Number'] = lowestGame
results.to_csv(YEAR + "results" + segment + ".csv")


