import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score


#Setting up the base directory and the model settings
BASE_DIR = Path(__file__).resolve().parent.parent #Going up two folders from the current script to reach the project root
STATS = ["GF", "GA", "SF", "SA", "SFT", "SAT"] #Six stats per team that will be used to train the model
MAX_GOALS = 10 #Largest number of goals per team we calculate probabilities for
OFFSEASON = False #Set to True if the match you are predicting is the first of a new season (applies the same summer Elo regression as training)
RUN_VALIDATION = False #Set to False to skip the walk-forward validation and start predicting faster
TUNE_ALPHA = True #Set to True to test several regularization strengths and automatically use the best one
ALPHA = 0.1 #Regularization strength of the Poisson models (only used as-is when TUNE_ALPHA is False)
WF_SEASONS = 5 #Number of most recent seasons used as walk-forward test seasons

#Loading the processed matches dataset
df = pd.read_csv(BASE_DIR / "Data" / "matches_features.csv") #Reading the processed features CSV into a dataframe
df["Date"] = pd.to_datetime(df["Date"]) #Converting the Date column into a datetime object
df = df.sort_values(["Date", "Time"]).reset_index(drop=True) #Sorting all matches by date and resetting the index
df["Season"] = np.where(df["Date"].dt.month >= 8, df["Date"].dt.year, df["Date"].dt.year - 1) #Labeling each match with the year its season started (a season starts in August) If the match is before August, it is part of the season in the previous calendar year, and if it is in August or after, it is part of the current calendar year's season. For example, a match on 2023-09-15 would be labeled as part of the 2023 season, while a match on 2024-05-10 would also be labeled as part of the 2023 season.

#Calculating the difference between Home and Away averages (Home minus Away)
for stat in STATS: #Looping through each of the 6 stats
    df[f"Diff_{stat}_Last15"] = df[f"Home_{stat}_Last15"] - df[f"Away_{stat}_Last15"] #Home minus Away difference for this stat

#Listing all feature columns that the Poisson models will use to predict expected goals
feature_cols = ["Home_Elo", "Away_Elo", "Diff_Elo"] #Pre-match Elo ratings and their difference
for stat in STATS: #Looping through each of the 6 stats
    feature_cols += [f"Home_{stat}_Last15", f"Away_{stat}_Last15", f"Diff_{stat}_Last15"] #Adding the home, away and difference version of each stat


#Function to build a Poisson regression model (scaling is needed so the regularization treats every feature fairly)
def build_poisson(alpha=1.0):
    return make_pipeline(
        StandardScaler(), #Standardizing every feature so they are all on the same scale
        PoissonRegressor(alpha=alpha, max_iter=1000) #Poisson regression with an L2 penalty (alpha) to stop overfitting on correlated features
    )


#Function to turn two expected goal values into a scoreline grid and Away / Draw / Home probabilities
def scoreline_grid(lam_home, lam_away):
    goals = np.arange(0, MAX_GOALS + 1) #Array of possible goal counts: 0, 1, 2, ... MAX_GOALS
    home_pmf = poisson.pmf(goals, lam_home) #Probability of the home team scoring exactly 0, 1, 2... goals
    away_pmf = poisson.pmf(goals, lam_away) #Probability of the away team scoring exactly 0, 1, 2... goals
    matrix = np.outer(home_pmf, away_pmf) #Scoreline grid where rows = home goals and columns = away goals (assumes the two teams score independently)
    matrix = matrix / matrix.sum() #Re-normalizing so the grid sums to 1 after cutting off at MAX_GOALS
    probs = np.array([np.triu(matrix, 1).sum(), np.trace(matrix), np.tril(matrix, -1).sum()]) #Adding up [Away Win, Draw, Home Win] (same order as ResultCode)
    return matrix, probs #Returning the scoreline grid and the outcome probabilities


#Function to run walk-forward validation (train on all earlier seasons, test on the next one, then move forward)
def walk_forward(alpha, n_test_seasons=WF_SEASONS):
    test_seasons = sorted(df["Season"].unique())[-n_test_seasons:] #The most recent seasons that will each be used as a test season
    rows = [] #Creating an empty list to store the scores of each test season
    for s in test_seasons: #Looping through each test season in chronological order
        train, test = df[df["Season"] < s], df[df["Season"] == s] #Training on every earlier season and testing on this one (the model never sees the future)
        hm = build_poisson(alpha).fit(train[feature_cols], train["FTHG"]) #Learning to predict home goals from past seasons only
        am = build_poisson(alpha).fit(train[feature_cols], train["FTAG"]) #Learning to predict away goals from past seasons only
        probs = np.array([scoreline_grid(h, a)[1] for h, a in zip(hm.predict(test[feature_cols]), am.predict(test[feature_cols]))]) #Outcome probabilities [Away, Draw, Home] for each test match
        base_rates = train["ResultCode"].value_counts(normalize=True).reindex([0, 1, 2]).values #How often away wins, draws and home wins happened in the training seasons
        base_probs = np.tile(base_rates, (len(test), 1)) #Baseline that predicts those same rates for every match
        rows.append({
            "Season": f"{s}-{str(s + 1)[-2:]}", #Season label like 2023-24
            "Matches": len(test), #Number of test matches in this season
            "LogLoss": log_loss(test["ResultCode"], probs, labels=[0, 1, 2]), #Log loss of the Poisson model (lower is better)
            "BaseLogLoss": log_loss(test["ResultCode"], base_probs, labels=[0, 1, 2]), #Log loss of always guessing the base rates (the score to beat)
            "Accuracy": accuracy_score(test["ResultCode"], probs.argmax(axis=1)), #How often the most likely outcome was correct
            "HomeWinBase": (test["ResultCode"] == 2).mean() #How often "always guess home win" would be right
        })
    return pd.DataFrame(rows) #Returning one row per test season


#Running the walk-forward validation (and optionally tuning the regularization strength)
best_alpha = ALPHA #Starting with the alpha chosen at the top of the file
if RUN_VALIDATION: #Checking if the user wants validation
    if TUNE_ALPHA: #Checking if the user wants to search for the best alpha
        tuning = {} #Creating an empty dictionary to store the average log loss of each alpha
        for a in [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0]: #Looping through the regularization strengths we want to compare
            tuning[a] = walk_forward(a)["LogLoss"].mean() #Saving the average walk-forward log loss for this alpha
            print(f"alpha={a:<6} average log loss: {tuning[a]:.4f}") #Printing the score so the user can see the trend
        best_alpha = min(tuning, key=tuning.get) #Picking the alpha with the lowest log loss
        print(f"Best alpha: {best_alpha}") #Showing which alpha was chosen
    wf = walk_forward(best_alpha) #Running the final walk-forward validation with the chosen alpha
    print("\n=== Walk-forward validation ===") #Printing the header
    print(wf.round(4).to_string(index=False)) #Printing the scores for every test season
    print(f"\nAverage log loss: {wf['LogLoss'].mean():.4f} (base rates: {wf['BaseLogLoss'].mean():.4f}, lower is better)") #Comparing the model to the baseline
    print(f"Average accuracy: {wf['Accuracy'].mean():.1%}") #How often the most likely outcome was correct on average

#Training the final models on 100% of historical matches
home_model = build_poisson(best_alpha).fit(df[feature_cols], df["FTHG"]) #Learning to predict how many goals the home team scores
away_model = build_poisson(best_alpha).fit(df[feature_cols], df["FTAG"]) #Learning to predict how many goals the away team scores

#Loading the final Elo ratings saved by the feature script so prediction Elo matches training Elo
with open(BASE_DIR / "Data" / "elo_state.json") as f: #Opening the Elo file created at the end of the feature script
    elo_state = json.load(f)["elo"] #Dictionary of {team name: most recent Elo rating}


#Function to grab a team's most up-to-date Elo rating and last 15 games of stats
def get_latest_team_state(team_name):
    team_matches = df[(df["HomeTeam"] == team_name) | (df["AwayTeam"] == team_name)] #Filtering all matches where this team played either home or away
    if team_matches.empty: #Checking if the team name was misspelled or does not exist in the dataset
        raise ValueError(f"Team '{team_name}' not found. Available teams: {sorted(df['HomeTeam'].unique())}") #Raising an error with a list of valid team names

    records = [] #Creating an empty list to store each match's stats from this team's perspective
    for row in team_matches.itertuples(): #Looping through every match this team has played in chronological order
        if row.HomeTeam == team_name: #Checking if the team played at home in this row
            records.append({"GF": row.FTHG, "GA": row.FTAG, "SF": row.HS, "SA": row.AS, "SFT": row.HST, "SAT": row.AST}) #Saving home match stats
        else: #If the team played away in this row
            records.append({"GF": row.FTAG, "GA": row.FTHG, "SF": row.AS, "SA": row.HS, "SFT": row.AST, "SAT": row.HST}) #Saving away match stats

    hist = pd.DataFrame(records) #Turning the team's match history list into a dataframe
    elo = elo_state.get(team_name, 1450.0) #Grabbing the team's current Elo from the saved file (1450 if it is missing)
    if OFFSEASON: #Checking if we are predicting the first match after the summer break
        elo = 0.75 * elo + 0.25 * 1500.0 #Pulling the rating 25% toward 1500, exactly like the training loop does
    state = {"Elo": elo} #Starting the team's state dictionary with its Elo rating
    for stat in STATS: #Looping through each of the 6 core stats
        state[f"{stat}_Last15"] = hist[stat].tail(15).mean() #Averaging the last 15 games for each stat
    return state #Returning the dictionary containing the team's current Elo and 15-game averages


#Function to predict the outcome probabilities and most likely scorelines of an upcoming match
def predict_match(home_team, away_team):
    h = get_latest_team_state(home_team) #Fetching the latest stats and Elo for the home team
    a = get_latest_team_state(away_team) #Fetching the latest stats and Elo for the away team

    row = {"Home_Elo": h["Elo"], "Away_Elo": a["Elo"], "Diff_Elo": h["Elo"] - a["Elo"]} #Starting the matchup feature dictionary with both Elo ratings and their difference
    for stat in STATS: #Looping through the 6 rolling stats to populate home, away, and difference features
        row[f"Home_{stat}_Last15"] = h[f"{stat}_Last15"] #Adding home team's 15-game rolling stat
        row[f"Away_{stat}_Last15"] = a[f"{stat}_Last15"] #Adding away team's 15-game rolling stat
        row[f"Diff_{stat}_Last15"] = h[f"{stat}_Last15"] - a[f"{stat}_Last15"] #Calculating Home minus Away difference for this stat

    X_match = pd.DataFrame([row])[feature_cols] #Converting the single-match dictionary into a dataframe with the exact column order used in training
    lam_home = home_model.predict(X_match)[0] #Predicting the home team's expected goals
    lam_away = away_model.predict(X_match)[0] #Predicting the away team's expected goals
    matrix, probs = scoreline_grid(lam_home, lam_away) #Converting expected goals into a scoreline grid and [Away Win, Draw, Home Win] probabilities
    top = np.dstack(np.unravel_index(np.argsort(matrix.ravel())[::-1][:3], matrix.shape))[0] #Finding the 3 highest-probability scorelines as (home goals, away goals)

    print(f"\n=== {home_team} (Elo: {h['Elo']:.0f}) vs {away_team} (Elo: {a['Elo']:.0f}) ===") #Printing the matchup header with current Elo ratings
    print(f"Expected goals: {home_team} {lam_home:.2f} - {lam_away:.2f} {away_team}") #Printing the expected goals for each side
    print(f"Home Win ({home_team}): {probs[2]:.1%}") #Printing the predicted probability of a Home Win (class 2)
    print(f"Draw:                 {probs[1]:.1%}") #Printing the predicted probability of a Draw (class 1)
    print(f"Away Win ({away_team}): {probs[0]:.1%}") #Printing the predicted probability of an Away Win (class 0)
    print("Most likely scores: " + ", ".join(f"{i}-{j} ({matrix[i, j]:.1%})" for i, j in top)) #Printing the 3 most likely exact scorelines
    return {"home_win": probs[2], "draw": probs[1], "away_win": probs[0], "exp_home_goals": lam_home, "exp_away_goals": lam_away} #Returning the numbers too so other code can use them


#Function to ask the user for a team name and keep asking until it is valid
def pick_team(prompt, teams):
    while True: #Repeating until the user types a valid team
        name = input(prompt).strip() #Reading the team name the user typed
        match = [t for t in teams if t.lower() == name.lower()] #Matching the name regardless of capital letters
        if match: #Checking if the team exists in the dataset
            return match[0] #Returning the correctly spelled team name
        print("Team not found, check the spelling and try again.") #Telling the user the name was not recognized


#Letting the user choose any matchup
recent = df[df["Date"] >= df["Date"].max() - pd.Timedelta(days=400)] #Matches from roughly the last season, used to show current teams
current_teams = sorted(set(recent["HomeTeam"]) | set(recent["AwayTeam"])) #Teams that have played recently
all_teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"])) #Every team in the dataset (older teams can still be picked)
print("\nCurrent teams:", ", ".join(current_teams)) #Showing the user which team names they can type

while True: #Letting the user predict as many matchups as they want
    home = pick_team("\nHome team: ", all_teams) #Asking for the home team
    away = pick_team("Away team: ", all_teams) #Asking for the away team
    if home == away: #Checking that the user did not pick the same team twice
        print("Pick two different teams.") #Asking the user to try again
        continue #Going back to the start of the loop
    predict_match(home, away) #Predicting the chosen matchup
    if input("\nAnother match? (y/n): ").strip().lower() != "y": #Asking if the user wants to keep going
        break #Stopping the loop when the user types anything other than y
