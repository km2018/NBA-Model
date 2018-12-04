import time

from nba_py import league, constants, team
from requests import get

import numpy as np
import pandas as pd

def findStat(teamID, data):
'''
    Parameters:
        teamID: the ID for the team of interest

        data: a csv file containing stats for each team

    Returns:
        a numpy array with the stats for the specific team from the season data csv
'''
    
    indices = (data.index.values).astype(np.int64)
    for r in range(0, len(indices)):
        if(teamID == indices[r]):
            return np.array((data.iloc[r, :]))
            break

def getGames(season):
'''
    Parameters:
        season: the season for the games

    Returns:
        a pandas Dataframe containing the games for specified season
'''
    TEAM_ID_COLUMN = 1
    GAME_MATCHUP_COLUMN = 6
    GAME_RESULT_COLUMN = 7
    DATAFRAME_COLUMNS = ['B_W/L', 'C_hostID', 'D_visitingID']

    season_gamelog = []
    game_numbers = []

    winloss_to_binary = {'W': 1, 'L': 0}

    gamelog_object = league.GameLog(season = season, player_or_team = 'T')
    gamelog = gamelog_object.overall()
    sorted_games = gamelog.sort_values(by = ['GAME_ID'], ascending = True)

    for i in range(len(sorted_games)):
        
        game_data = []

        # Skip every other entry
        # Every 2 entries represents a game so this must be done
        if (i % 2 == 1 or i == len(sorted_games) - 1):
            continue

        matchup = str(sorted_games.iloc[i, GAME_MATCHUP_COLUMN])

        # If vs. is in matchup sorted_games[i] is the home team
        if "vs." not in matchup:
            binary_result = winloss_to_binary[sorted_games.iloc[i + 1, GAME_RESULT_COLUMN]]
            game_data.append(binary_result)

            home_id = int(sorted_games.iloc[i + 1, GAME_RESULT_COLUMN])
            road_id = int(sorted_games.iloc[i, GAME_RESULT_COLUMN])

            game_data.append(home_id)
            game_data.append(road_id)
        
        # If vs. is not in matchup sorted_games[i] is the road team
        else:
            binary_result = winloss_to_binary[sorted_games.iloc[i, GAME_RESULT_COLUMN]]
            game_data.append(binary_result)

            home_id = int(sorted_games.iloc[i, GAME_RESULT_COLUMN])
            road_id = int(sorted_games.iloc[i + 1, GAME_RESULT_COLUMN])

            game_data.append(home_id)
            game_data.append(road_id)

        season_gamelog.append(game_data)
        game_numbers.append(int((i / 2) + 1))

    season_gamelog_dataframe = pd.DataFrame(data = season_gamelog, index = game_numbers, columns = DATAFRAME_COLUMNS)
    return season_gamelog_dataframe

def getTeamPIE(year, min_games_played = 45, min_mins_played = 25, num_players_considered = 2):
'''
    Parameters:
        year: the season/year of interest

        min_games_played: the minimum amount of games played to be qualified as
            the top players on the teams to contribute the Team PIE.
            Default is 45 games.

        mins_played: the minimum amount of minutes played per game to be
            qualified as the top players on the teams to contribute the Team PIE.
            Default is 25 minutes.

        num_players_considered: by default, the function gets the two players with the highest PIE
            for each team, but the number of players considered can be changed by changing
            num_players

    Returns:
        a pandas DataFrame containing the total Player Impact Estimate (PIE)
        of the two highest ranked players for each team
'''
    DESIRED_COLUMNS = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'PIE']

    advanced_player_stats = league.PlayerStats(measure_type = "Advanced", season = year)
    all_advanced_stats = advanced_player_stats.overall()
    sorted_stats = all_advanced_stats.sort_values(by = ['PIE'], ascending = False)
    
    unqualified_players = []
    team_ids = []
    team_pies = []

    for current_player in range(len(sorted_stats)):
        if(sorted_stats['GP'][current_player] <= min_games_played and sorted_stats['MIN'][current_player] <= mins_played):
            unqualified_players.append(current_player)

    sorted_stats = sorted_stats.drop(unqualified_players)
    sorted_stats = sorted_stats[DESIRED_COLUMNS]

    for team in constants.TEAMS:
        team_ids.append(constants.TEAMS[team]['id'])

    for team_id in team_ids:
        counter = 0
        team_pie = 0

        for index, row in sorted_stats.iterrows():

            if(counter == num_players_considered):
                team_pies.append(team_pie)
                break

            if(int(row['TEAM_ID']) == int(team_id)):
                team_pie += row['PIE']
                counter += 1

    team_pie_dataframe = pd.DataFrame(data = team_pies, index = team_ids, columns = ['Team PIE'])
    return(team_pie_dataframe)

def getDF(season, measure_type):
'''
    Parameters:
        season: the season of interest
        measure_type: 'Four Factors' or 'Advanced'
    Returns:
        a pandas DataFrame containing all teams' specified type of stats
'''
    FOUR_FACTORS_END_COLUMN = 15
    ADVANCED_END_COLUMN = 9

    URL = "https://stats.nba.com/stats/leaguedashteamstats"
    HEADERS = {
        'user-agent': ('Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36'),  # noqa: E501
        'Dnt': ('1'),
        'Accept-Encoding': ('gzip, deflate, sdch'),
        'Accept-Language': ('en'),
        'origin': ('http://stats.nba.com')
        }
    DICT_HEADERS = dict(HEADERS)

    PARAMS = {
        'Conference': '',
        'DateFrom': '',
        'DateTo': '',
        'Division': '',
        'GameScope': '',
        'GameSegment': '',
        'LastNGames': '0',
        'LeagueID': "00",
        'Location': '',
        'MeasureType': measureType,
        'Month': '0',
        'OpponentTeamID': '0',
        'Outcome': '',
        'PORound': '0',
        'PaceAdjust': "N",
        'PerMode': "PerGame",
        'Period': '0',
        'PlayerExperience': '',
        'PlayerPosition': '',
        'PlusMinus': "N",
        'Rank': "N",
        'Season': season,
        'SeasonSegment': '',
        'SeasonType': "Regular Season",
        'ShotClockRange': '',
        'StarterBench': '',
        'TeamID': '0',
        'VsConference': '',
        'VsDivision': ''
    }

    _get = get(URL, params = PARAMS, headers = DICT_HEADERS)
    data = _get.json()
    headers = data['resultSets'][0]['headers']
    rows = data['resultSets'][0]['rowSet']

    indices = []
    season_data = []
    end_column = 0

    if(measure_type == "Four Factors"):
        end_column = FOUR_FACTORS_END_COLUMN
    elif(measure_type == "Advanced"):
        end_column = ADVANCED_END_COLUMN

    for row in rows:
        indices.append(row[0])
        season_data.append(row[7:end_column])

    np_season_data = np.array(season_data)
    season_data_dataframe = pd.DataFrame(data = np_season_data, columns = headers[7:end_column], index = indices)
    return (season_data_dataframe)

def getTeamLocationDF(team_id, season):
'''
    Parameters:
        team_id: ID for the team of intersst
        season: season of interest
    Returns:
        a pandas DataFrame containing the team's four factors
        based on the location of game (home or away)
'''

    URL = "https://stats.nba.com/stats/teamdashboardbygeneralsplits"
    HEADERS = {
        'user-agent': ('Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36'),  # noqa: E501
        'Dnt': ('1'),
        'Accept-Encoding': ('gzip, deflate, sdch'),
        'Accept-Language': ('en'),
        'origin': ('http://stats.nba.com')
        }
    DICT_HEADERS = dict(HEADERS)

    PARAMS = {
        'Conference': '',
        'DateFrom': '',
        'DateTo': '',
        'Division': '',
        'GameScope': '',
        'GameSegment': '',
        'LastNGames': '0',
        'LeagueID': "00",
        'Location': '',
        'MeasureType': 'Four Factors',
        'Month': '0',
        'OpponentTeamID': '0',
        'Outcome': '',
        'PORound': '0',
        'PaceAdjust': "N",
        'PerMode': "PerGame",
        'Period': '0',
        'PlayerExperience': '',
        'PlayerPosition': '',
        'PlusMinus': "N",
        'Rank': "N",
        'Season': season,
        'SeasonSegment': '',
        'SeasonType': "Regular Season",
        'ShotClockRange': '',
        'StarterBench': '',
        'TeamID': team_id,
        'VsConference': '',
        'VsDivision': ''
    }

    _get = get(url, params = PARAMS, headers = DICT_HEADERS)
    season_data = _get.json()
    headers = season_data['resultSets'][1]['headers']
    values = season_data['resultSets'][1]['rowSet']

    return (pd.DataFrame(data = values, columns = headers))

def getTeamCourt(team_id, season, court_type):
'''
    Parameters:
        team_id: Id of the team of interest
        season: season of interest
        court_type: 'home' or 'road'
    Returns:
        an array containing the team's court data
'''
    
    court = getTeamLocationDF(team_id = team_id, season = season)
    if(courtType == "home"):
        i = 0

    elif(courtType == "road"):
        i = 1

    return [court['EFG_PCT'][i], court['FTA_RATE'][i], court['TM_TOV_PCT'][i],
            court['OREB_PCT'][i], court['OPP_EFG_PCT'][i], court["OPP_FTA_RATE"][i],
            court['OPP_TOV_PCT'][i], court['OPP_OREB_PCT'][i]]

def getCourtData(season):
'''
    Requires:
        teamData.csv under current working directory
    Parameters:
        season: season of interest
    Returns:
        a pandas DataFrame containing all teams' court data for the season
'''

    METRICS = ["Home_EFG_PCT", "Home_FTA_RATE", "Home_TM_TOV_PCT", "Home_OREB_PCT",
    "Home_OPP_EFG_PCT","Home_OPP_FTA_RATE","Home_OPP_TOV_PCT","Home_OPP_OREB_PCT",
    "Road_EFG_PCT","Road_FTA_RATE","Road_TM_TOV_PCT","Road_OREB_PCT","Road_OPP_EFG_PCT",
    "Road_OPP_FTA_RATE","Road_OPP_TOV_PCT", "Road_OPP_OREB_PCT"]
    TIME_BETWEEN_FUNCTION_CALLS = 2

    team_data = pd.DataFrame.from_csv("teamData.csv")
    team_ids = teamData.index.values

    court_data = []

    for i in range(len(team_ids)):
        team_court_data = []
        home_court_data = getTeamCourt(team_id = team_ids[i], season = season, courtType = "home")
        time.sleep(TIME_BETWEEN_FUNCTION_CALLS)
        road_court_data = getTeamCourt(team_id = team_ids[i], season = season, courtType = "road")

        [team_court_data.append(home_court_data[i]) for i in range(len(home_court_data))]
        [team_court_data.append(road_court_data[i]) for i in range(len(road_court_data))]
        court_data.append(team_court_data)

    court_data_dataframe = pd.DataFrame(data = court_data, columns = METRICS, index = team_ids)
    return court_data_dataframe

def getSeasonDF(season):
'''
    Requires:
        2010-17data.csv under the current working directory
    Parameters:
        season: the season of interest
    Returns:
        a pandas DataFrame containing all team stats of current
        interest for the given season
'''
    
    gameLogs = getGames(season)
    ff = getDF(season, "Four Factors")
    rtg = getDF(season, "Advanced")
    pie = getTeamPIE(season)
    courtData = getCourtData(season)
    sample = pd.DataFrame.from_csv("2010-17data.csv")
    v = sample.columns.values
    c = courtData.columns.values

    column = []
    for i in gameLogs.columns.values:
        column.append(i)
    for i in ff.columns.values:
        column.append(i + "_Diff")
    for i in rtg.columns.values:
        column.append(i + "_Diff")
    column.append("PIE_Diff")
    [column.append(v[14 + z]) for z in range(8)]

    values = []

    for r in range(len(gameLogs)):
        print(r)
        temp = []
        homeID = gameLogs.iloc[r, 1]
        roadID = gameLogs.iloc[r, 2]

        for i in gameLogs.iloc[r, :]:
            temp.append(i)
        homeFFs = findStat(homeID, ff)
        roadFFs = findStat(roadID, ff)

        for f in range(len(homeFFs)):
            temp.append((homeFFs[f] - roadFFs[f]) * 100)

        homeRTGs = findStat(homeID, rtg)
        roadRTGs = findStat(roadID, rtg)
        for g in range(len(homeRTGs)):
            temp.append(homeRTGs[g] - roadRTGs[g])

        homePIE = findStat(homeID, pie)
        roadPIE = findStat(roadID, pie)
        temp.append((homePIE[0] - roadPIE[0]) * 100)

        [temp.append((courtData[c[m]][homeID] -
                  courtData[c[m + 8]][roadID]) * 100)
                  for m in range(8)]

        values.append(temp)

    bam = pd.DataFrame(data=values, columns=column)
    return bam

# basically nothing to do
def getSeasonsDF(startYear, endYear):
'''
    Parameters:
        startYear: year to start at (inclusive)
        endYear: year to end (exclusive)
    Returns:
        a pandas DataFrame containing all team stats from
        startYear to endYear
'''

    seasonsDF = pd.concat([getSeasonDF(str(i) + "-" + str((i + 1))[2:])
                           for i in range(startYear, endYear)])
    return seasonsDF
