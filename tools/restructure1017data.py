from nba_py import constants, league, team
import numpy as np
import pandas as pd
import time

def restructureIndices(csv):
	intermediate = pd.read_csv(csv)
	tempindices = intermediate["Unnamed: 0"]
	indices = list(tempindices)
	cleanedDF = intermediate.drop(columns = ["Unnamed: 0"])
	cleanedDF.index = indices 
	return(cleanedDF)

data2010 = restructureIndices("2010-11CourtData.csv")
data2011 = restructureIndices("2011-12CourtData.csv")
data2012 = restructureIndices("2012-13CourtData.csv")
data2013 = restructureIndices("2013-14CourtData.csv")
data2014 = restructureIndices("2014-15CourtData.csv")
data2015 = restructureIndices("2015-16CourtData.csv")
data2016 = restructureIndices("2016-17CourtData.csv")

tempagg = pd.read_csv("2005-2017data.csv")
aggregate_data = tempagg[:][6150:]
homeIDs = list(aggregate_data["C_hostID"])
roadIDs = list(aggregate_data["D_visitingID"])

courtEFGPCT = []
courtFTA = []
courtTOVPCT = []
courtOREBPCT = []
oppcourtEFGPCT = []
oppcourtFTA = []
oppcourtTOVPCT = []
oppcourtOREBPCT = []


for i in range(0, 1230):
	courtEFGPCT.append(data2010["Home_EFG_PCT"][homeIDs[i]] - data2010["Road_EFG_PCT"][roadIDs[i]])
	courtFTA.append(data2010["Home_FTA_RATE"][homeIDs[i]] - data2010["Road_FTA_RATE"][roadIDs[i]])
	courtTOVPCT.append(data2010["Home_TM_TOV_PCT"][homeIDs[i]] - data2010["Road_TM_TOV_PCT"][roadIDs[i]])
	courtOREBPCT.append(data2010["Home_OREB_PCT"][homeIDs[i]] - data2010["Road_OREB_PCT"][roadIDs[i]])
	oppcourtEFGPCT.append(data2010["Home_OPP_EFG_PCT"][homeIDs[i]] - data2010["Road_OPP_EFG_PCT"][roadIDs[i]])
	oppcourtFTA.append(data2010["Home_OPP_FTA_RATE"][homeIDs[i]] - data2010["Road_OPP_FTA_RATE"][roadIDs[i]])
	oppcourtTOVPCT.append(data2010["Home_OPP_TOV_PCT"][homeIDs[i]] - data2010["Road_OPP_TOV_PCT"][roadIDs[i]])
	oppcourtOREBPCT.append(data2010["Home_OPP_OREB_PCT"][homeIDs[i]] - data2010["Road_OPP_OREB_PCT"][roadIDs[i]])
for i in range(1230, 2460):
	courtEFGPCT.append(data2011["Home_EFG_PCT"][homeIDs[i]] - data2011["Road_EFG_PCT"][roadIDs[i]])
	courtFTA.append(data2011["Home_FTA_RATE"][homeIDs[i]] - data2011["Road_FTA_RATE"][roadIDs[i]])
	courtTOVPCT.append(data2011["Home_TM_TOV_PCT"][homeIDs[i]] - data2011["Road_TM_TOV_PCT"][roadIDs[i]])
	courtOREBPCT.append(data2011["Home_OREB_PCT"][homeIDs[i]] - data2011["Road_OREB_PCT"][roadIDs[i]])
	oppcourtEFGPCT.append(data2011["Home_OPP_EFG_PCT"][homeIDs[i]] - data2011["Road_OPP_EFG_PCT"][roadIDs[i]])
	oppcourtFTA.append(data2011["Home_OPP_FTA_RATE"][homeIDs[i]] - data2011["Road_OPP_FTA_RATE"][roadIDs[i]])
	oppcourtTOVPCT.append(data2011["Home_OPP_TOV_PCT"][homeIDs[i]] - data2011["Road_OPP_TOV_PCT"][roadIDs[i]])
	oppcourtOREBPCT.append(data2011["Home_OPP_OREB_PCT"][homeIDs[i]] - data2011["Road_OPP_OREB_PCT"][roadIDs[i]])
for i in range(2460, 3690):
	courtEFGPCT.append(data2012["Home_EFG_PCT"][homeIDs[i]] - data2012["Road_EFG_PCT"][roadIDs[i]])
	courtFTA.append(data2012["Home_FTA_RATE"][homeIDs[i]] - data2012["Road_FTA_RATE"][roadIDs[i]])
	courtTOVPCT.append(data2012["Home_TM_TOV_PCT"][homeIDs[i]] - data2012["Road_TM_TOV_PCT"][roadIDs[i]])
	courtOREBPCT.append(data2012["Home_OREB_PCT"][homeIDs[i]] - data2012["Road_OREB_PCT"][roadIDs[i]])
	oppcourtEFGPCT.append(data2012["Home_OPP_EFG_PCT"][homeIDs[i]] - data2012["Road_OPP_EFG_PCT"][roadIDs[i]])
	oppcourtFTA.append(data2012["Home_OPP_FTA_RATE"][homeIDs[i]] - data2012["Road_OPP_FTA_RATE"][roadIDs[i]])
	oppcourtTOVPCT.append(data2012["Home_OPP_TOV_PCT"][homeIDs[i]] - data2012["Road_OPP_TOV_PCT"][roadIDs[i]])
	oppcourtOREBPCT.append(data2012["Home_OPP_OREB_PCT"][homeIDs[i]] - data2012["Road_OPP_OREB_PCT"][roadIDs[i]])
for i in range(3690, 4920):
	courtEFGPCT.append(data2013["Home_EFG_PCT"][homeIDs[i]] - data2013["Road_EFG_PCT"][roadIDs[i]])
	courtFTA.append(data2013["Home_FTA_RATE"][homeIDs[i]] - data2013["Road_FTA_RATE"][roadIDs[i]])
	courtTOVPCT.append(data2013["Home_TM_TOV_PCT"][homeIDs[i]] - data2013["Road_TM_TOV_PCT"][roadIDs[i]])
	courtOREBPCT.append(data2013["Home_OREB_PCT"][homeIDs[i]] - data2013["Road_OREB_PCT"][roadIDs[i]])
	oppcourtEFGPCT.append(data2013["Home_OPP_EFG_PCT"][homeIDs[i]] - data2013["Road_OPP_EFG_PCT"][roadIDs[i]])
	oppcourtFTA.append(data2013["Home_OPP_FTA_RATE"][homeIDs[i]] - data2013["Road_OPP_FTA_RATE"][roadIDs[i]])
	oppcourtTOVPCT.append(data2013["Home_OPP_TOV_PCT"][homeIDs[i]] - data2013["Road_OPP_TOV_PCT"][roadIDs[i]])
	oppcourtOREBPCT.append(data2013["Home_OPP_OREB_PCT"][homeIDs[i]] - data2013["Road_OPP_OREB_PCT"][roadIDs[i]])
for i in range(4920, 6150):
	courtEFGPCT.append(data2014["Home_EFG_PCT"][homeIDs[i]] - data2014["Road_EFG_PCT"][roadIDs[i]])
	courtFTA.append(data2014["Home_FTA_RATE"][homeIDs[i]] - data2014["Road_FTA_RATE"][roadIDs[i]])
	courtTOVPCT.append(data2014["Home_TM_TOV_PCT"][homeIDs[i]] - data2014["Road_TM_TOV_PCT"][roadIDs[i]])
	courtOREBPCT.append(data2014["Home_OREB_PCT"][homeIDs[i]] - data2014["Road_OREB_PCT"][roadIDs[i]])
	oppcourtEFGPCT.append(data2014["Home_OPP_EFG_PCT"][homeIDs[i]] - data2014["Road_OPP_EFG_PCT"][roadIDs[i]])
	oppcourtFTA.append(data2014["Home_OPP_FTA_RATE"][homeIDs[i]] - data2014["Road_OPP_FTA_RATE"][roadIDs[i]])
	oppcourtTOVPCT.append(data2014["Home_OPP_TOV_PCT"][homeIDs[i]] - data2014["Road_OPP_TOV_PCT"][roadIDs[i]])
	oppcourtOREBPCT.append(data2014["Home_OPP_OREB_PCT"][homeIDs[i]] - data2014["Road_OPP_OREB_PCT"][roadIDs[i]])
for i in range(6150, 7380):
	courtEFGPCT.append(data2015["Home_EFG_PCT"][homeIDs[i]] - data2015["Road_EFG_PCT"][roadIDs[i]])
	courtFTA.append(data2015["Home_FTA_RATE"][homeIDs[i]] - data2015["Road_FTA_RATE"][roadIDs[i]])
	courtTOVPCT.append(data2015["Home_TM_TOV_PCT"][homeIDs[i]] - data2015["Road_TM_TOV_PCT"][roadIDs[i]])
	courtOREBPCT.append(data2015["Home_OREB_PCT"][homeIDs[i]] - data2015["Road_OREB_PCT"][roadIDs[i]])
	oppcourtEFGPCT.append(data2015["Home_OPP_EFG_PCT"][homeIDs[i]] - data2015["Road_OPP_EFG_PCT"][roadIDs[i]])
	oppcourtFTA.append(data2015["Home_OPP_FTA_RATE"][homeIDs[i]] - data2015["Road_OPP_FTA_RATE"][roadIDs[i]])
	oppcourtTOVPCT.append(data2015["Home_OPP_TOV_PCT"][homeIDs[i]] - data2015["Road_OPP_TOV_PCT"][roadIDs[i]])
	oppcourtOREBPCT.append(data2015["Home_OPP_OREB_PCT"][homeIDs[i]] - data2015["Road_OPP_OREB_PCT"][roadIDs[i]])
for i in range(7380, len(aggregate_data)):
	courtEFGPCT.append(data2016["Home_EFG_PCT"][homeIDs[i]] - data2016["Road_EFG_PCT"][roadIDs[i]])
	courtFTA.append(data2016["Home_FTA_RATE"][homeIDs[i]] - data2016["Road_FTA_RATE"][roadIDs[i]])
	courtTOVPCT.append(data2016["Home_TM_TOV_PCT"][homeIDs[i]] - data2016["Road_TM_TOV_PCT"][roadIDs[i]])
	courtOREBPCT.append(data2016["Home_OREB_PCT"][homeIDs[i]] - data2016["Road_OREB_PCT"][roadIDs[i]])
	oppcourtEFGPCT.append(data2016["Home_OPP_EFG_PCT"][homeIDs[i]] - data2016["Road_OPP_EFG_PCT"][roadIDs[i]])
	oppcourtFTA.append(data2016["Home_OPP_FTA_RATE"][homeIDs[i]] - data2016["Road_OPP_FTA_RATE"][roadIDs[i]])
	oppcourtTOVPCT.append(data2016["Home_OPP_TOV_PCT"][homeIDs[i]] - data2016["Road_OPP_TOV_PCT"][roadIDs[i]])
	oppcourtOREBPCT.append(data2016["Home_OPP_OREB_PCT"][homeIDs[i]] - data2016["Road_OPP_OREB_PCT"][roadIDs[i]])

aggregate_data["court_EFG_PCT_DIFF"] = courtEFGPCT
aggregate_data["court_FTA_RATE_DIFF"] = courtFTA
aggregate_data["court_TOV_PCT_DIFF"] = courtTOVPCT
aggregate_data["court_OREB_PCT_DIFF"] = courtOREBPCT
aggregate_data["opp_court_EFG_PCT_DIFF"] = oppcourtEFGPCT
aggregate_data["opp_court_FTA_RATE_DIFF"] = oppcourtFTA
aggregate_data["opp_court_TOV_PCT_DIFF"] = oppcourtTOVPCT
aggregate_data["opp_court_OREB_PCT_DIFF"] = oppcourtOREBPCT



aggregate_data.to_csv("final2010-17data.csv")