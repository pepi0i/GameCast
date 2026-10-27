import pandas as pd

df = pd.read_csv('Data/epl_18-19.csv')

print(df.head)

#Read all the files from data/ titled epl
#read each file into a table and stack them into a list
#Stack the list into a single table(making sure to reset the index)
#Keep only 7 columns: Date, Time, HomeTeam, AwayTeam, FTHG, FTAG, and FTR
     #Drop any columns that are not needed to make the new table with only the 7 columns
#check shape to make sure the new table has the correct number of rows and columns
#turn the date column into a datetime object, and now sort the tables by date, and then reset the index
#Assign numeric values to the match results, ie. Home Win = 2, Draw = 1, Away Win = 0
    #Using .map() function, create a new column called Result and assign the numeric values to the match results
#Create a match id by using the index of the table and assign it to a new column called MatchID


#Create a table for Home Team stats, using the Date, Time, HomeTeam, FTHG, FTAG, and match_id columns, but rename the columns to Date, Time, Team, GoalsFor, GoalsAgainst, and MatchID
#Do the same for Away Team stats, but rename the columns to Date, Time, Team, GoalsFor, GoalsAgainst, and MatchID
      #For the away team stats, make sure to switch the GoalsFor and GoalsAgainst columns, as the away team is the one that scored the goals against the home team
#Combine the home and away team stats into a single table (team_games), and sort by team, then by date, and then finally reset the index
#Create a new column for the table, called GF_Last5, which will be the mean of the GoalsFor column for the last 5 matches for each team, and assign it to the new column
       #In doing so, make sure to group by team, but also to shift, that way you do not include the current match in the calculation of the mean for the last 5 matches
      #Do the same for the GoalsAgainst column, and create a new column called GA_Last5

#Drop the rows in team_games except for Team, GF_Last5, GA_Last5, and MatchID
#Create a new table called home_team_games, which will be a copy of team_games, but only for the home team games, and rename the columns to HomeTeam, GF_Last5, GA_Last5, and MatchID
#Do the same for the away team games, and rename the columns to AwayTeam, GF
#Using the original combined table of all the matches, merge it with home_team_games on the MatchID column, and then merge it with away_team_games on the MatchID column
#Check the shape of the final table to make sure it has the correct number of rows and columns, and then save it to a csv file called epl_processed.csv
