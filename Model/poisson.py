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
ALPHA = 0.1 #Regularization strength of the Poisson models
WF_SEASONS = 5 #Number of most recent seasons used as walk-forward test seasons


#Model state that is initialized when load_model() runs
feature_cols = []
df = pd.DataFrame()
home_model = None
away_model = None
elo_state = {}




def build_poisson(alpha=1.0):
   return make_pipeline(
       StandardScaler(),
       PoissonRegressor(alpha=alpha, max_iter=1000)
   )


#Function that calculates the probabilty of each scoreline using the expected goals for each team, and then returns a matrix of the scorelines
def scoreline_grid(lam_home, lam_away):
   goals = np.arange(0, MAX_GOALS + 1)
   home_pmf = poisson.pmf(goals, lam_home)
   away_pmf = poisson.pmf(goals, lam_away)
   matrix = np.outer(home_pmf, away_pmf)
   matrix = matrix / matrix.sum()
   probs = np.array([np.triu(matrix, 1).sum(), np.trace(matrix), np.tril(matrix, -1).sum()])
   return matrix, probs




#Function that loads the match data, trains the Poisson models, and loads the Elo ratings from a JSON file
#Loading all of this once is neccesary for the app, because the models and Elo ratings are used for every prediction, and loading them each time would be inefficient
def load_model():
   global df, feature_cols, home_model, away_model, elo_state


   df = pd.read_csv(BASE_DIR / "Data" / "matches_features.csv")
   df["Date"] = pd.to_datetime(df["Date"])
   df = df.sort_values(["Date", "Time"]).reset_index(drop=True)
   df["Season"] = np.where(df["Date"].dt.month >= 8, df["Date"].dt.year, df["Date"].dt.year - 1)


   for stat in STATS:
       df[f"Diff_{stat}_Last15"] = df[f"Home_{stat}_Last15"] - df[f"Away_{stat}_Last15"]


   feature_cols = ["Home_Elo", "Away_Elo", "Diff_Elo"]
   for stat in STATS:
       feature_cols += [f"Home_{stat}_Last15", f"Away_{stat}_Last15", f"Diff_{stat}_Last15"]


   home_model = build_poisson(ALPHA).fit(df[feature_cols], df["FTHG"])
   away_model = build_poisson(ALPHA).fit(df[feature_cols], df["FTAG"])


   with open(BASE_DIR / "Data" / "elo_state.json") as f:
       elo_state = json.load(f)["elo"]


   return {"df": df, "feature_cols": feature_cols, "home_model": home_model, "away_model": away_model, "elo_state": elo_state, "current_teams:": sorted(df["HomeTeam"].unique())}




#Function to grab a team's most up-to-date Elo rating and last 15 games of stats
def get_latest_team_state(team_name, model):
    df = model["df"]
    elo_state = model["elo_state"]

    team_matches = df[(df["HomeTeam"] == team_name) | (df["AwayTeam"] == team_name)]
    if team_matches.empty:
       raise ValueError(f"Team '{team_name}' not found. Available teams: {sorted(df['HomeTeam'].unique())}")


    records = []
    for row in team_matches.itertuples():
       if row.HomeTeam == team_name:
           records.append({"GF": row.FTHG, "GA": row.FTAG, "SF": row.HS, "SA": row.AS, "SFT": row.HST, "SAT": row.AST})
       else:
           records.append({"GF": row.FTAG, "GA": row.FTHG, "SF": row.AS, "SA": row.HS, "SFT": row.AST, "SAT": row.HST})


    hist = pd.DataFrame(records)
    elo = elo_state.get(team_name, 1450.0)
    if OFFSEASON:
       elo = 0.75 * elo + 0.25 * 1500.0
    state = {"Elo": elo}
    for stat in STATS:
       state[f"{stat}_Last15"] = hist[stat].tail(15).mean()
    return state




#Function to predict the outcome probabilities and most likely scorelines of an upcoming match
def predict_match(home_team, away_team):
   h = get_latest_team_state(home_team)
   a = get_latest_team_state(away_team)


   row = {"Home_Elo": h["Elo"], "Away_Elo": a["Elo"], "Diff_Elo": h["Elo"] - a["Elo"]}
   for stat in STATS:
       row[f"Home_{stat}_Last15"] = h[f"{stat}_Last15"]
       row[f"Away_{stat}_Last15"] = a[f"{stat}_Last15"]
       row[f"Diff_{stat}_Last15"] = h[f"{stat}_Last15"] - a[f"{stat}_Last15"]


   X_match = pd.DataFrame([row])[feature_cols]
   lam_home = home_model.predict(X_match)[0]
   lam_away = away_model.predict(X_match)[0]
   matrix, probs = scoreline_grid(lam_home, lam_away)
   top = np.dstack(np.unravel_index(np.argsort(matrix.ravel())[::-1][:3], matrix.shape))[0]


   print(f"\n=== {home_team} (Elo: {h['Elo']:.0f}) vs {away_team} (Elo: {a['Elo']:.0f}) ===")
   print(f"Expected goals: {home_team} {lam_home:.2f} - {lam_away:.2f} {away_team}")
   print(f"Home Win ({home_team}): {probs[2]:.1%}")
   print(f"Draw:                 {probs[1]:.1%}")
   print(f"Away Win ({away_team}): {probs[0]:.1%}")
   print("Most likely scores: " + ", ".join(f"{i}-{j} ({matrix[i, j]:.1%})" for i, j in top))
   return {"home_win": probs[2], "draw": probs[1], "away_win": probs[0], "exp_home_goals": lam_home, "exp_away_goals": lam_away}









