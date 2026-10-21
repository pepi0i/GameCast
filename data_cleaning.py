import glob
import pandas as pd

files = sorted(glob.glob("data/epl_*.csv"))
frames = [pd.read_csv(f) for f in files]
matches = pd.concat(frames, ignore_index=True)

print(matches.shape)

cols = ["Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
matches = matches[cols]
matches = matches.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])

print(matches.shape)
print(matches.head())