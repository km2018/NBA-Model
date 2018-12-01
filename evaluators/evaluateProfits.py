import warnings

from keras import optimizers
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.models import load_model
import time
import numpy as np
import pandas as pd

# Calculates profit from bet given moneyline and wager size
def getProfit(moneyline, wager_size = 100):
    if(moneyline > 0):
        return moneyline
    else:
        return abs((wager_size / moneyline) * 100)

warnings.filterwarnings("ignore", category = np.VisibleDeprecationWarning)

# Constants
YEAR = '2015'
DATA_FILE_NAME = YEAR + "data.csv"
MODEL_FILE_NAME = YEAR + "model.h5"
RESULTS_FILE_NAME = YEAR + "results.csv"
UNUSED_COLUMN = ["Unnamed: 0"]
COLUMN_AXIS = 1

FEATURES_START_COLUMN = 3
NUM_FEATURES = 19
FEATURES_END_COLUMN = FEATURES_START_COLUMN + NUM_FEATURES
HOME_LINE_COLUMN = FEATURES_END_COLUMN
ROAD_LINE_COLUMN = HOME_LINE_COLUMN + 1
GAME_RESULT_COLUMN = 0

NUM_GAMES = 1230

EV_STEP_SIZE = 4
MIN_EV_FLOOR = 0
MAX_EV_FLOOR = 16 + EV_STEP_SIZE

PROB_WINDOW_STEP_SIZE = 10
MIN_LOWER_PROB = 5
MAX_LOWER_PROB = 45 + PROB_WINDOW_STEP_SIZE

# Read in input data and model
input_data = pd.read_csv(DATA_FILE_NAME).drop(UNUSED_COLUMN, axis = COLUMN_AXIS)
model = load_model(MODEL_FILE_NAME)

# Create lists for metrics we want to track
expected_values = []
lower_probability_bounds = []
upper_probability_bounds = []
season_profits = []
bet_percentages = []
accuracy = []
returns = []
lowest_bankrolls = []
games_of_lowest_bankroll = []


for ev_floor in range(MIN_EV_FLOOR, MAX_EV_FLOOR, EV_STEP_SIZE):                                     
    
    for lower_prob in range (MIN_LOWER_PROB, MAX_LOWER_PROB, PROB_WINDOW_STEP_SIZE):
        
        # Intialize metrics we want to track
        lower_prob_bound = lower_prob / 100
        upper_prob_bound = 1 - lower_prob_bound

        season_profit = 0
        lowest_bankroll = 0
        lowest_bankroll_game = 0
        games_betted = 0

        num_correct = 0

        for game_number in range (0, len(input_data)):

            # Skip bet on game if no money line exists
            if(input_data.iloc[game_number, HOME_LINE_COLUMN] == "NL" or input_data.iloc[game_number, ROAD_LINE_COLUMN] == "NL"):
                continue

            # Get money lines and result of game
            home_line = int(input_data.iloc[game_number, HOME_LINE_COLUMN])
            road_line = int(input_data.iloc[game_number, ROAD_LINE_COLUMN])
            game_result = input_data.iloc[game_number, GAME_RESULT_COLUMN]

            # Get feature data for game and make prediction
            input_list = np.array(input_data.iloc[game_number, FEATURES_START_COLUMN : FEATURES_END_COLUMN])
            input_list.resize((1, NUM_FEATURES))
            raw_prediction = model.predict(input_list)[0][0]            

            # Skip bet on game if not within probability window
            if(raw_prediction < lower_prob_bound or raw_prediction > upper_prob_bound):
                continue

            # Expected Value = (money from winning * likelihood of winning) - (loss * probability of loss)
            home_expected_value = (getProfit(home_line) * raw_prediction) - (100 * (1 - raw_prediction))
            road_expected_value = (getProfit(road_line) * (1 - raw_prediction)) - (100 * raw_prediction)
            
            # Skip bet on game if neither expected value is high enough
            if(home_expected_value < ev_floor and road_expected_value < ev_floor):
                continue

            # Get prediction based on which expected value is higher
            prediction = int(home_expected_value > road_expected_value)
            game_profit = -100

            # Calculate profit from game if prediction was correct
            if(prediction == game_result):
                if prediction == 1:
                    game_profit = getProfit(home_line)
                
                elif prediction == 0:
                    game_profit = getProfit(road_line)
                
                num_correct += 1

            games_betted += 1
            season_profit += game_profit

            if(season_profit < lowest_bankroll):
                lowest_bankroll = season_profit
                lowest_bankroll_game = game_number + 1

        # Add metrics calculated to lists
        expected_values.append(ev_floor)
        lower_probability_bounds.append(lower_prob_bound)
        upper_probability_bounds.append(upper_prob_bound)

        season_profits.append(season_profit)
        returns.append(season_profit / games_betted)
        bet_percentages.append(games_betted / NUM_GAMES)
        accuracy.append(num_correct / games_betted)
        
        lowest_bankrolls.append(lowest_bankroll)
        games_of_lowest_bankroll.append(lowest_bankroll_game)

        # Print metrics for current ev floor and probability window configuration
        print("Expected value floor: %d" % (ev_floor))
        print("Lower bound: %f" % (lower_prob_bound))
        print("Upper bound: %f" % (upper_prob_bound))
        print("Total Profit for 2016 season is: %.2f" % (season_profit))
        print("Average profit is: %.2f" % (season_profit / games_betted))
        print("Percentage of Games Betted: %.2f" % (games_betted / NUM_GAMES))
        print("Percentage of Games Correct: %.2f" % (num_correct / games_betted))
        print("\n")

# Store calculated metrics in dataframe for all ev floor and probability window configurations
# Write out dataframe to specified CSV file
results = pd.DataFrame(data = expected_values, columns = ['Expected Value Floor'])
results['Lower Probability Bound'] = lower_probability_bounds
results['Upper Probability Bound'] = upper_probability_bounds
results['Profit for Season'] = season_profits
results['Return'] = returns
results['Percent of Games Betted'] = bet_percentages
results['Accuracy'] = accuracy
results['Lowest Bankroll for Season'] = lowest_bankrolls
results['Game of Lowest Bankroll'] = games_of_lowest_bankroll
results.to_csv(RESULTS_FILE_NAME)


