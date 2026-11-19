import sys
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.append(str(Path(__file__).resolve().parent.parent))
from Model.poisson import load_model, predict_match #Importing the two functions the page needs




# Load model once and cache it
@st.cache_resource(show_spinner="Loading and training the model...")
def get_model():
   return load_model()




model = get_model()


# Get available teams
teams = sorted(model["df"]["HomeTeam"].dropna().unique())




st.title("GameCast Match Predictor")


st.write(
   "This project predicts match outcomes based on previous results"
)

if "history" not in st.session_state:
    st.session_state.history = [] 

left, right = st.columns(2)
with left: 
   home_team =  st.selectbox("Home Team", teams)
with right: 
   away_team = st.selectbox("Away Team", teams, index=1)
   
if home_team == away_team:
   st.warning("Please pick two different teams!")


else:
   st.success(f"Valid Matchup: {home_team} vs {away_team}")




if st.button("Predict Match", icon=":material/bolt:", width="stretch", type="primary", disabled=(home_team==away_team)):


   # Predict the match
   result = predict_match(home_team, away_team, model) 
   now = datetime.now(ZoneInfo("America/Los_Angeles"))  # Get current time in Pacific Time Zone
   st.session_state.history.append({
        "Time": now.strftime("%b %d, %Y %I:%M:%S %p %Z"),
        "Match": f"{home_team} vs {away_team}",
        "Home win": f"{result['home_win']:.1%}",
        "Draw": f"{result['draw']:.1%}",
        "Away win": f"{result['away_win']:.1%}",
    })

   # Display warnings
   for note in result["notes"]:
       st.warning(note)


   # Display probabilities
   c1, c2, c3 = st.columns(3)


   c1.metric(f"{home_team} win", f"{result['home_win']:.1%}")
   c2.metric("Draw", f"{result['draw']:.1%}")
   c3.metric(f"{away_team} win", f"{result['away_win']:.1%}")


   # Probability chart
   chart_data = pd.DataFrame(
       {
           "Probability": [
               result["home_win"],
               result["draw"],
               result["away_win"]
           ]
       },
       index=[
           f"{home_team} win",
           "Draw",
           f"{away_team} win"
       ]
   )


   st.bar_chart(chart_data)


   # Expected goals
   st.write(
       f"**Expected goals:** {home_team} "
       f"{result['exp_home_goals']:.2f} - "
       f"{result['exp_away_goals']:.2f} {away_team}"
   )


   # Elo ratings
   st.write(
       f"**Elo ratings:** {home_team} "
       f"{result['home_elo']:.0f} | "
       f"{away_team} {result['away_elo']:.0f}"
   )


   # Most likely scores
   st.write(
       "**Most likely scores:** " +
       ", ".join(
           f"{s['score']} ({s['prob']:.1%})"
           for s in result["top_scores"]
       )
   )

   st.write(
         f"**Prediction Made at:** {now.strftime('%Y-%m-%d %H:%M:%S %p %Z')}"
   )
if st.session_state.history:
    st.subheader("Your predictions this session")
    st.dataframe(pd.DataFrame(st.session_state.history[::-1]), hide_index=True)

