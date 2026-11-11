import pandas as pd
import glob
from pathlib import Path






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
columns = ["Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
combined = combined[columns]
combined = combined.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])




#Turning the Date column into a datetime object and sorting the data by date
combined["Date"] = pd.to_datetime(combined["Date"], format="%d/%m/%Y", errors="coerce")
combined = combined.sort_values("Date")
combined = combined.reset_index(drop=True)


#Assigning numeric values to match results
#Home Win = 2, Draw = 1, Away Win = 0
match_result = {"H": 2, "D": 1, "A": 0}
combined["ResultCode"] = combined["FTR"].map(match_result)
print(combined.head())


combined["match_id"] = combined.index #Creating a new column for match_id which is the index of the dataframe


print(files)


#Table for Home Teams
home_team = pd.DataFrame({
   "Date" : combined["Date"],
   "Time" : combined["Time"],
   "Team" : combined["HomeTeam"],
   "GF" : combined["FTHG"],
   "GA" : combined["FTAG"],
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
   "match_id" : combined.index
})
print(away_team.head())


#Combining the home and away team tables into one table for all teams
team_games = pd.concat([home_team, away_team], ignore_index=True)
team_games = team_games.sort_values(["Team", "Date"]) #Sort by team and date to get the games in order for each team
team_games = team_games.reset_index(drop=True) #Resetting the index after sorting


print(team_games.head(10))


team_games["GF_Last5"] = team_games.groupby("Team")["GF"].shift(1).rolling(5).mean() #Creating a new column for the average goals scored in the last 5 games for each team
team_games["GA_Last5"] = team_games.groupby("Team")["GA"].shift(1).rolling(5).mean() #Creating a new column for the average goals conceded in the last 5 games for each team
#We shift one back to not include the current game in the average, avoiding data leakage


print(team_games.head(10))


print(combined.head())


cols = ["Team", "GF_Last5", "GA_Last5", "match_id"]
team_games = team_games[cols]
team_games = team_games.dropna(subset=["Team", "GF_Last5", "GA_Last5", "match_id"]) #Dropping rows with missing values in the specified columns


home_team_games = team_games.rename(columns={"Team" : "HomeTeam", "GF_Last5" : "Home_GF_Last5", "GA_Last5" : "Home_GA_Last5"}) #Renaming the columns to match the combined table
away_team_games = team_games.rename(columns={"Team" : "AwayTeam", "GF_Last5" : "Away_GF_Last5", "GA_Last5" : "Away_GA_Last5"}) #Renaming the columns to match the combined table




combined = combined.merge(home_team_games, on=["match_id", "HomeTeam"]) #Merging the home_team_games table with the combined table on the match_id column
combined = combined.merge(away_team_games, on=["match_id", "AwayTeam"]) #Merging the away_team_games table with the combined table on the match_id column


print(combined.head(10))



from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
combined.to_csv(BASE_DIR / "Data" / "matches_features.csv", index=False)





