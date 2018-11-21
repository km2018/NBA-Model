import sys
sys.path.insert(0, '/Users/Karthik/Documents/GitHub/NBA-Model')

from nba_py_api import constants, league
import numpy as np
import pandas as pd

nametoID = {}
for team in constants.TEAMS:
	city = constants.TEAMS[team]['city'].replace(" ", "")
	team_id = constants.TEAMS[team]['id']

	if(city == "LosAngeles" and team_id == '1610612746'):
		city = "LAClippers"

	if(city == "LosAngeles" and team_id == '1610612747'):
		city = "LALakers"

	nametoID[city] = team_id

YEAR = '2017'
YEAR_RANGE = '1718'
COLUMN_AXIS = 1
TEAMS_PER_GAME = 2

ODDS_FILE = YEAR_RANGE + "odds.csv"
GAMEDATA_FILE = YEAR + "data.csv"
AUTO_GENERATED_COLUMN = ["Unnamed: 0"]
TEAM_COLUMN_ODDS = "Team"
HOST_ID_COLUMN_GAME = "C_hostID"
ROAD_ID_COLUMN_GAME = "D_visitingID"
MONEY_LINE_COLUMN = "ML"

HOME_LINE_COLUMN_NAME = "HOME_LINE"
ROAD_LINE_COLUMN_NAME = "ROAD_LINE"
OUTPUT_FILE = YEAR + "testing-data.csv"

odds_data = pd.read_csv(ODDS_FILE)
game_data = pd.read_csv(GAMEDATA_FILE).drop(AUTO_GENERATED_COLUMN, axis = COLUMN_AXIS)

money_lines = odds_data[MONEY_LINE_COLUMN]
team_names_odds_data = oddsData[TEAM_COLUMN_ODDS]
home_team_ids_game_data = game_data[HOST_ID_COLUMN_GAME]
road_team_ids_game_data = game_data[ROAD_ID_COLUMN_GAME]

odds_team_id_pairs = []
game_team_id_pairs = []

per_game_money_lines = []
home_lines = []
road_lines = []

for i in range(1, len(teamNames), TEAMS_PER_GAME):

	game_lines = []
	home_id = nametoID[teamNames[i].replace(" ", "")]
	road_id = nametoID[teamNames[i - 1].replace(" ", "")]
	
	odds_team_id_pairs.append(home_id + road_id)
	game_lines.append(money[i])
	game_lines.append(money[i - 1])
	per_game_money_lines.append(game_lines)

for i in range(0, len(host_team_ids_game_data)):

	home_id = home_team_ids_game_data[i]
	road_id = road_team_ids_game_data[i]
	game_team_id_pairs.append(home_id + road_id)

for game in game_team_id_pairs:

	true_game_num = sums.index(game)
	home_lines.append(lines[true_game_num][0])
	road_lines.append(lines[true_game_num][1]) 

	lines.pop(index)
	sums.remove(game)

game_data[HOME_LINE_COLUMN_NAME] = home_lines
game_data[ROAD_LINE_COLUMN_NAME] = road_lines
game_data.to_csv(OUTPUT_FILE)