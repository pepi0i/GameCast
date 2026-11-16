import pandas as pd
import glob
from pathlib import Path
import numpy as np








files = sorted((Path(__file__).resolve().parent.parent / "Data").glob("epl*.csv"))








data_table = []


#Reading each of the csv files and appending them to one list
for file in files:
   df = pd.read_csv(file)
  
   data_table.append(df)




print(len(data_table))




#combining all the dataframes into one and reset the index
combined = pd.concat(data_table, ignore_index=True)


#Dropping unnecessary columns and rows with missing values
#These are the only columns that we will take into account for our predictions
columns = ["Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HS", "AS", "HST", "AST",]
combined = combined[columns]
combined = combined.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HS", "AS", "HST", "AST"]) #Dropping rows with missing values in the specified columns




#Turning the Date column into a datetime object and sorting the data by date
combined["Date"] = pd.to_datetime(combined["Date"], format="%d/%m/%Y", errors="coerce")
combined = combined.sort_values("Date")
combined = combined.reset_index(drop=True)


#Assigning numeric values to match results
#Home Win = 2, Draw = 1, Away Win = 0
match_result = {"H": 2, "D": 1, "A": 0}
combined["ResultCode"] = combined["FTR"].map(match_result)
print(combined.head())


elo_ratings = {}
home_elos = []
away_elos = []

K = 20          # Base learning rate
HOME_ADV = 65   # Premier League home-field Elo boost
prev_date = None

for row in combined.itertuples():
    # Pull ratings 25% toward 1500 across the summer off-season (>45 days gap)
    if prev_date is not None and (row.Date - prev_date).days > 45:
        for team in elo_ratings:
            elo_ratings[team] = 0.75 * elo_ratings[team] + 0.25 * 1500.0
    prev_date = row.Date

    h_team = row.HomeTeam
    a_team = row.AwayTeam

    # Newly promoted/unseen teams start slightly below average (1450)
    h_elo = elo_ratings.get(h_team, 1450.0)
    a_elo = elo_ratings.get(a_team, 1450.0)

    # Save pre-match Elo BEFORE updating (zero data leakage)
    home_elos.append(h_elo)
    away_elos.append(a_elo)

    # Expected score for Home and Away
    exp_home = 1.0 / (1.0 + 10.0 ** ((a_elo - (h_elo + HOME_ADV)) / 400.0))
    exp_away = 1.0 - exp_home

    # Actual match score
    if row.FTR == "H":
        actual_home, actual_away = 1.0, 0.0
    elif row.FTR == "D":
        actual_home, actual_away = 0.5, 0.5
    else:
        actual_home, actual_away = 0.0, 1.0

    # Goal-difference multiplier (rewards dominant wins, keeps draws at 1.0)
    goal_diff = abs(row.FTHG - row.FTAG)
    mov_multiplier = np.log(goal_diff + 1) if goal_diff > 0 else 1.0

    # Update ratings after the match
    elo_ratings[h_team] = h_elo + (K * mov_multiplier) * (actual_home - exp_home)
    elo_ratings[a_team] = a_elo + (K * mov_multiplier) * (actual_away - exp_away)

combined["Home_Elo"] = home_elos
combined["Away_Elo"] = away_elos
combined["Diff_Elo"] = combined["Home_Elo"] - combined["Away_Elo"]



combined["match_id"] = combined.index #Creating a new column for match_id which is the index of the dataframe


print(files)


#Table for Home Teams
home_team = pd.DataFrame({
   "Date" : combined["Date"],
   "Time" : combined["Time"],
   "Team" : combined["HomeTeam"],
   "GF" : combined["FTHG"],
   "GA" : combined["FTAG"],
   "Shots_For" : combined["HS"],
   "Shots_For_on_Target" : combined["HST"],
   "Shots_Against" : combined["AS"],
   "Shots_Against_on_Target" : combined["AST"],
   "match_id" : combined.index
})
print(home_team.head())


#Table for Away Teams
away_team = pd.DataFrame({
   "Date" : combined["Date"],
   "Time" : combined["Time"],
   "Team" : combined["AwayTeam"],
   "GF" : combined["FTAG"],
   "GA" : combined["FTHG"],
   "Shots_For" : combined["AS"],
   "Shots_For_on_Target" : combined["AST"],
   "Shots_Against" : combined["HS"],
   "Shots_Against_on_Target" : combined["HST"],
   "match_id" : combined.index
})
print(away_team.head())


#Combining the home and away team tables into one table for all teams
team_games = pd.concat([home_team, away_team], ignore_index=True)
team_games = team_games.sort_values(["Team", "Date"]) #Sort by team and date to get the games in order for each team
team_games = team_games.reset_index(drop=True) #Resetting the index after sorting


print(team_games.head(10))


team_games["GF_Last15"] = team_games.groupby("Team")["GF"].shift(1).rolling(15).mean() #Creating a new column for the average goals scored in the last 5 games for each team
team_games["GA_Last15"] = team_games.groupby("Team")["GA"].shift(1).rolling(15).mean() #Creating a new column for the average goals conceded in the last 5 games for each team
#We shift one back to not include the current game in the average, avoiding data leakage

team_games["SF_Last15"] = team_games.groupby("Team")["Shots_For"].shift(1).rolling(15).mean() #Creating a new column for the average shots for in the last 5 games for each team
team_games["SA_Last15"] = team_games.groupby("Team")["Shots_Against"].shift(1).rolling(15).mean() #Creating a new column for the average shots against in the last 5 games for each team
#We shift one back to not include the current game in the average, avoiding data leakage

team_games["SFT_Last15"] = team_games.groupby("Team")["Shots_For_on_Target"].shift(1).rolling(15).mean() #Creating a new column for the average shots on target for in the last 5 games for each team
team_games["SAT_Last15"] = team_games.groupby("Team")["Shots_Against_on_Target"].shift(1).rolling(15).mean() #Creating a new column for the average shots on target against in the last 5 games for each team
#We shift one back to not include the current game in the average, avoiding data leakage

print(team_games.head(10))


print(combined.head())


cols = ["Team", "GF_Last15", "GA_Last15", "SF_Last15", "SA_Last15", "SFT_Last15", "SAT_Last15", "match_id"]
team_games = team_games[cols]
team_games = team_games.dropna(subset=["Team", "GF_Last15", "GA_Last15", "SF_Last15", "SA_Last15", "SFT_Last15", "SAT_Last15", "match_id"]) #Dropping rows with missing values in the specified columns


home_team_games = team_games.rename(columns={"Team" : "HomeTeam", "GF_Last15" : "Home_GF_Last15", "GA_Last15" : "Home_GA_Last15", "SF_Last15" : "Home_SF_Last15", "SA_Last15" : "Home_SA_Last15", "SFT_Last15" : "Home_SFT_Last15", "SAT_Last15" : "Home_SAT_Last15"}) #Renaming the columns to match the combined table
away_team_games = team_games.rename(columns={"Team" : "AwayTeam", "GF_Last15" : "Away_GF_Last15", "GA_Last15" : "Away_GA_Last15", "SF_Last15" : "Away_SF_Last15", "SA_Last15" : "Away_SA_Last15", "SFT_Last15" : "Away_SFT_Last15", "SAT_Last15" : "Away_SAT_Last15"}) #Renaming the columns to match the combined table




combined = combined.merge(home_team_games, on=["match_id", "HomeTeam"]) #Merging the home_team_games table with the combined table on the match_id column
combined = combined.merge(away_team_games, on=["match_id", "AwayTeam"]) #Merging the away_team_games table with the combined table on the match_id column


print(combined.head(10))



from pathlib import Path

#Lets the user know where the file is being saved
BASE_DIR = Path(__file__).resolve().parent.parent
combined.to_csv(BASE_DIR / "Data" / "matches_features.csv", index=False) #Turning the combined dataframe into a csv file and saving it in the Data folder

import json #Used to save the final Elo ratings to a file
with open(BASE_DIR / "Data" / "elo_state.json", "w") as f: #Creating the Elo file in the Data folder
    json.dump({"elo": elo_ratings}, f) #Saving the most recent Elo rating for every team




