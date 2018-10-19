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


#warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
# data = pd.read_csv("bettingData16-17.csv")
#data = np.loadtxt("bettingData16-17.csv", dtype=str, delimiter=",")

# IDs of the two teams you are predicting
# host_ID = '1610612759'
# road_ID = '1610612753'
#

# pX = predict[1:, 4:]
# pY = predict[1:, 1]

# create model
model = Sequential()
# model = load_model("2016model.h5")
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

# maxProfit = 0
# EVs = []
# probWindows = []
# tProfits = []
# avgProfits = []
# percentGamesBetted = []
# percentCorrect = []
# #=========================================================================
# # Looping through different expected value floors, which are the minimum expected values required for us
# # to bet.
# #=========================================================================
# for m in range(40):
#     EVFloor = -100 + m * 5
#     print("------------------------------\nEXPECTED VALUE FLOOR IS: %d" % (EVFloor))
# 
#     for j in range(40):
#         totalProfit = 0
#         gamesBetted = 0
#         correctGames = 0
#         predLower = 0.05 + j / 100  # lower bound of probability window
#         predUpper = 0.95 - j / 100  # upper bound of probability window
#         
#         #    The probability window is where we bet when the
#         #    model's prediction falls into it. In this loop,
#         #    the window shrinks from 5-95% to 45-55% in decrement of 1% on both bounds
#         #    The idea is that heavy underdogs/favorites yield
#         #    very little profits, so we only want to bet when the model
#         #    isn't quite certain which team will win.
#         
#         windowSize = (predUpper - predLower) * 100
# 
#         probWindows.append(windowSize)
#         EVs.append(EVFloor)
# 
#         for i in range(1, len(data)):
#             # Feeding the input parameters into the model
#             inputList = np.array(data[i, 4:18])
#             inputList.resize((1, 14))
# 
#             mPred = model.predict(inputList)[0][0]
#             pred = round(mPred)
#             profit = -100
#             gameResult = data[i, 1]
#                 
#             if(str(data[i, 18]) == "NL" or str(data[i, 19]) == "NL"):
#                 continue
#             
#             homeline = int(str(data[i, 18]))
#             roadline = int(str(data[i, 19]))
#             
#             # Calculating the expected values for each team
#             homeEV = getProfit(homeline) * mPred - (100 * (1 - mPred))
#             roadEV = getProfit(roadline) * (1 - mPred) - (100 * mPred)
# 
#             # We don't bet if the expected value is lower than the floor, or if the model's prediction
#             # is outside of the window we set
#             if(homeEV < EVFloor and roadEV < EVFloor or (mPred < predLower or mPred > predUpper)):
#                 continue
#             
#             pred = int(homeEV > roadEV)
#             
#             if(pred and gameResult):
#                 profit = getProfit(homeline)
#                 correctGames += 1
#             elif(not pred and not gameResult):
#                 profit = getProfit(roadline)
#                 correctGames += 1
#             else:
#                 profit = -100
# 
#             gamesBetted += 1
# 
#             totalProfit += profit
# 
#         tProfits.append(totalProfit)
#         avgProfits.append(totalProfit / gamesBetted)
#         percentGamesBetted.append(gamesBetted / len(data))
#         percentCorrect.append(correctGames / gamesBetted)
#         if(totalProfit > maxProfit):
#             maxProfit = totalProfit
#         print("Lower bound: %f, Upper bound: %f Window Size: %d\nTotal Profit for 17-18 season is: %.2f \nAverage profit is: %.2f\n Percentage of Games Betted: %.2f%%\nPercentage of Games Correct: %.2f%%\n"
#               % (predLower, predUpper, windowSize, totalProfit, totalProfit / gamesBetted,
#                  (gamesBetted / len(data) * 100), (correctGames / gamesBetted) * 100))
#         
# print("Maximum total profit is: %f" % (maxProfit))
# 
# # Recording data from every iteration above into a CSV
# rows = []
# heads = ['EV', 'ProbWind', 'Return', 'Profits', 'percentGamesBetted', 'percentGamesCorrect']
# for p in range(len(EVs)):
#     temp = []
#     temp.append(EVs[p])
#     temp.append(probWindows[p])
#     temp.append(avgProfits[p])
#     temp.append(tProfits[p])
#     temp.append(percentGamesBetted[p])
#     temp.append(percentCorrect[p])
#     rows.append(temp)
# info = pd.DataFrame(data=rows, columns=heads)
# info.to_csv("2016modelML_EV_-100_to_100.csv")
# 
# # Plotting profits vs probability windows & average profits (aka percent returns since we're betting w/ $100)
# # in a 3D scatter plot
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# 
# ax.scatter(EVs, probWindows, avgProfits)
# ax.set_title('Profits vs EV & ProbWindows')
# ax.set_xlabel("Expected value floor")
# ax.set_ylabel("Prob Windows")
# ax.set_zlabel("Avg Profits")
# 
# plt.show()

# plt.scatter(EVs,tProfits)
# plt.title("Expected Value floors vs total profits")
# plt.xlabel("EVs")
# plt.ylabel("Profits")
# plt.show()
#
# plt.scatter(probWindows,tProfits)
# plt.title("Probability windows vs total profits")
# plt.xlabel("probability window size")
# plt.ylabel("Profits")
# plt.show()

# a = 0
# seasons =[]
# accs = []
# losses = []
# for i in range(12):
#     season = str(2005+i) + "-" + str(2006+i)[2:]
#     seasons.append(season)
#     X = np.array(data.iloc[1 + a: 1231 + a, 4:])
#     print (X.size)
#     X.resize((1230,14))
#     Y = np.array(data.iloc[1 + a: 1231 + a, 1])
#     scores = model.evaluate(X, Y,verbose=0)
#     losses.append(scores[0])
#     accs.append(scores[1])
#     print("\n%s: %.2f%%" % (model.metrics_names[0], scores[0] * 100))
#     print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
#
#     if(season == "2011-12"):
#         a += 990
#     else:
#         a += 1230
#
# plt.scatter(seasons,accs)
# plt.title("accuracy vs seasons")
# plt.xlabel("Seasons")
# plt.ylabel("Accuracy")
# plt.show()
#
# plt.scatter(losses,accs)
# plt.title("losses vs seasons")
# plt.xlabel("Seasons")
# plt.ylabel("Losses")
# plt.show()
# calculate predictions
# scores = model.evaluate(pX, pY)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
