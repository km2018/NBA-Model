from nba_py import constants, league, team
import numpy as np
import pandas as pd
import time

bigData = pd.read_csv("2010-17data.csv").drop(['Unnamed: 0'], axis = 1)

data2010 = bigData[:][:1230]
data2011 = bigData[:][1230:2220]
data2012 = bigData[:][2220:3450]
data2013 = bigData[:][3450:4680]
data2014 = bigData[:][4680:5910]
data2015 = bigData[:][5910:7140]
data2016 = bigData[:][7140:]

data2010.to_csv('2010data.csv')
data2011.to_csv('2011data.csv')
data2012.to_csv('2012data.csv')
data2013.to_csv('2013data.csv')
data2014.to_csv('2014data.csv')
data2015.to_csv('2015data.csv')
data2016.to_csv('2016data.csv')