from nba_py import constants, league
import numpy as np
import pandas as pd
import time

nametoID = {}
for team in constants.TEAMS:
	city = constants.TEAMS[team]['city'].replace(" ", "")
	ID = constants.TEAMS[team]['id']
	if(city == "LosAngeles" and ID == '1610612746'):
		city = "LAClippers"
	if(city == "LosAngeles" and ID == '1610612747'):
		city = "LALakers"
	nametoID[city] = ID

oddsData = pd.read_csv("1718odds.csv")
boxscoreData = pd.read_csv("2017data.csv")
#boxscoreData = temp[:1230]
#lists = [str(i) for i in range(0,1230)]
#boxscoreData.index = lists
teamNames = oddsData["Team"]
teamNames1 = boxscoreData["C_hostID"]
teamNames2 = boxscoreData["D_visitingID"]
money = oddsData["ML"]
sums = []
sums1 = []
lines = []
home_lines = []
road_lines = []
organized_lines = []
for i in range(1, len(teamNames)):
	if (i%2 == 0):
		continue
	temp = []
	sums.append(nametoID[teamNames[i].replace(" ", "")] + nametoID[teamNames[i-1].replace(" ", "")])
	temp.append(money[i])
	temp.append(money[i-1])
	lines.append(temp)
for i in range(0, len(teamNames1)):
	sums1.append(str(int(teamNames1[i])) + str(int(teamNames2[i])))
for game in sums1:
	index = sums.index(game)
	organized_lines.append(lines[index])
	lines.pop(index)
	sums.remove(game)
for line in organized_lines:
	home_lines.append(line[0])
	road_lines.append(line[1])
boxscoreData['HOME_LINE'] = home_lines
boxscoreData['ROAD_LINE'] = road_lines
fin = boxscoreData.drop(["Unnamed: 0"], axis = 1)
fin.to_csv("2017datafinal.csv")
#print(sums1.index("16106127471610612758"))
#print(lines.pop(sums1.index("16106127471610612758")))
#print(sums1.remove("16106127471610612758"))
#print(sums1.index("16106127471610612758"))
#print(lines[sums1.index("16106127471610612758")])
