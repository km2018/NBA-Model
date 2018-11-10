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

    compilation = []
    indices = []

    season = season
    binaryWL = {'W': 1, 'L': 0}

    games = league.GameLog(season=season, player_or_team='T')

    scores = games.overall()

    finScores = scores.sort_values(by=['GAME_ID'], ascending=True)
    
    for i in range(len(finScores)):
        temp = []
        if (i % 2 == 1 or i == len(finScores) - 1):
            continue
        matchup = str(finScores.iloc[i, 6])
        if "vs." not in matchup:
            WL = binaryWL[finScores.iloc[i + 1, 7]]
            temp.append(WL)
            hostID = int(finScores.iloc[i + 1, 1])
            roadID = int(finScores.iloc[i, 1])
            temp.append(hostID)
            temp.append(roadID)
        else:
            key = finScores.iloc[i, 7]
            WL = binaryWL[key]
            temp.append(WL)
            hostID1 = int(finScores.iloc[i, 1])
            roadID1 = int(finScores.iloc[i + 1, 1])
            temp.append(hostID1)
            temp.append(roadID1)
        compilation.append(temp)
        indices.append(int((i / 2) + 1))

    columns = ['B_W/L', 'C_hostID', 'D_visitingID']

    kaboom = pd.DataFrame(data=compilation, index=indices, columns=columns)
    return kaboom


def getTeamPIE(year):
    '''
    Parameters:
        year: the season/year of interest

    Returns: 
        a pandas DataFrame containing the Player Impact Estimate (PIE)
    of the two highest ranked players for each team
    '''

    obj = league.PlayerStats(measure_type="Advanced", season=year)
    df = obj.overall()
    PIE = df.sort_values(by=['PIE'], ascending=False)
    nonQualis = []
    teamIDs = []
    teamPIE = []

    for i in range(0, len(PIE)):
        if(PIE['GP'][i] <= 45 and PIE['MIN'][i] <= 25):
            nonQualis.append(i)
        else:
            pass
    PIE = PIE.drop(nonQualis)
    PIE = PIE[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'PIE']]
    for team in constants.TEAMS:
        teamIDs.append(constants.TEAMS[team]['id'])
    for team in teamIDs:
        temp = 0
        onePIE = 0
        for index, row in PIE.iterrows():
            if(temp == 2):
                teamPIE.append(onePIE)
                break
            if(int(row['TEAM_ID']) == int(team)):
                onePIE += row['PIE']
                temp += 1
    aggregate = pd.DataFrame(data=teamPIE, index=teamIDs, columns=['teamPIE'])
    return(aggregate)


def getDF(season, measureType):
    '''
    Parameters:
        season: the season of interest
        measureType: 'Four Factors' or 'Advanced'
    Returns:
        a pandas DataFrame containing all teams' specified type of stats
    '''
    url = "https://stats.nba.com/stats/leaguedashteamstats"
    HEADERS = {
        'user-agent': ('Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36'),  # noqa: E501
        'Dnt': ('1'),
        'Accept-Encoding': ('gzip, deflate, sdch'),
        'Accept-Language': ('en'),
        'origin': ('http://stats.nba.com')
        }
    headers = dict(HEADERS)

    params = {
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

    # 0, 6:
    # Index = teamIDs

    _get = get(url, params=params, headers=headers)
    data = _get.json()
    headers1 = data['resultSets'][0]['headers']
    rows1 = data['resultSets'][0]['rowSet']

    indices = []
    data = []
    end = 0

    if(measureType == "Four Factors"):
        end = 15
    elif(measureType == "Advanced"):
        end = 9

    for row in rows1:
        indices.append(row[0])
        data.append(row[7:end])

    total = np.array(data)
    df = pd.DataFrame(data=total, columns=headers1[7:end], index=indices)
    return (df)


def getTeamLocationDF(teamID, season):
    '''
    Parameters:
        teamID: ID for the team of intersst
        season: season of interest
    Returns:
        a pandas DataFrame containing the team's four factors 
        based on the location of game (home or away)
    '''

    url = "https://stats.nba.com/stats/teamdashboardbygeneralsplits"
    HEADERS = {
        'user-agent': ('Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36'),  # noqa: E501
        'Dnt': ('1'),
        'Accept-Encoding': ('gzip, deflate, sdch'),
        'Accept-Language': ('en'),
        'origin': ('http://stats.nba.com')
        }
    headers = dict(HEADERS)

    params = {
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
        'TeamID': teamID,
        'VsConference': '',
        'VsDivision': ''
    }

    _get = get(url, params=params, headers=headers)
    data = _get.json()
    headers1 = data['resultSets'][1]['headers']
    values = data['resultSets'][1]['rowSet']

    return (pd.DataFrame(data=values, columns=headers1))


def getTeamCourt(teamID, season, courtType):
    '''
    Parameters:
        teamID: Id of the team of interest
        season: season of interest
        courtType: 'home' or 'road'
    Returns:
        an array containing the team's court data
    '''
    court = getTeamLocationDF(teamID=teamID, season=season)
    if(courtType == "home"):
        i = 0
    elif(courtType == "road"):
        i = 1
    return [court['EFG_PCT'][i], court['FTA_RATE'][i], court['TM_TOV_PCT'][i],
            court['OREB_PCT'][i], court['OPP_EFG_PCT'][i], court["OPP_FTA_RATE"][i],
            court['OPP_TOV_PCT'][i], court['OPP_OREB_PCT'][i]]


def getCourtData(season):
    '''
    Parameters:
        season: season of interest
    Returns:
        a pandas DataFrame containing all teams' court data for the season
    '''

    sample = pd.DataFrame.from_csv("final2010-17data.csv")
    v = sample.columns.values

    teamData = pd.DataFrame.from_csv("teamData.csv")
    teamIds = teamData.index.values

    blah = []
    c = []

    [c.append("Home_" + v[8 + z]) for z in range(8)]
    [c.append("Road_" + v[8 + z]) for z in range(8)]
          
    for i in range(len(teamIds)):
        temp = []
        home = getTeamCourt(teamID=teamIds[i], season=season, courtType="home")
        time.sleep(2)
        road = getTeamCourt(teamID=teamIds[i], season=season, courtType="road")
        
        [temp.append(home[i]) for i in range(len(home))]
        [temp.append(road[i]) for i in range(len(road))]
        blah.append(temp)
        
    csv = pd.DataFrame(data=blah, columns=c, index=teamIds)
    return csv


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