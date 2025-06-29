import streamlit as st
import pandas as pd
import requests
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Dream11 Baseball AI", layout="centered", page_icon="⚾")
st.info("⏱ App loaded — refresh manually for now. Auto-refresh logic removed for compatibility.")

# =============================
# API Test
# =============================
def test_api_connection():
    url = "https://statsapi.mlb.com/api/v1/teams"
    response = requests.get(url)
    if response.status_code == 200:
        st.success("✅ Live API working!")
    else:
        st.error("❌ API connection failed.")

test_api_connection()

# =============================
# Simulate Match Data
# =============================
def get_upcoming_matches():
    return ["Dodgers vs Cubs", "Giants vs Braves"]

def get_confirmed_lineup(match):
    if "Dodgers" in match:
        return ["Mookie Betts", "Freddie Freeman", "Will Smith"]
    return []

def extract_opponent(match, player):
    team1, team2 = match.split(" vs ")
    if player in ["Mookie Betts", "Freddie Freeman", "Will Smith"]:
        return team2
    return team1

def get_weather_impact(match):
    if "Dodgers" in match:
        return 1.05
    elif "Giants" in match:
        return 0.95
    return 1.0

def get_opponent_difficulty(team):
    hard_teams = ["Braves", "Cubs"]
    return 0.95 if team in hard_teams else 1.0

def simulate_last_5_scores():
    return [random.randint(40, 90) for _ in range(5)]

def assign_player_role(player_name):
    return random.choice(["Batter", "Pitcher", "Outfielder"])

def fetch_player_stats(player_name, match=None):
    opponent = extract_opponent(match, player_name)
    role = assign_player_role(player_name)
    last_5 = simulate_last_5_scores()
    form_avg = sum(last_5) / len(last_5)
    risk = np.std(last_5)

    search_url = f"https://statsapi.mlb.com/api/v1/people/search?name={player_name.replace(' ', '%20')}"
    search_resp = requests.get(search_url)
    player_id = None

    try:
        results = search_resp.json().get("people", [])
        if results:
            player_id = results[0].get("id")
    except:
        pass

    avg = obp = hr = rbi = 0

    if player_id:
        stats_url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=season&group=hitting"
        stats_resp = requests.get(stats_url)
        try:
            stats = stats_resp.json()["stats"][0]["splits"][0]["stat"]
            avg = float(stats.get("avg", 0.0))
            obp = float(stats.get("obp", 0.0))
            hr = int(stats.get("homeRuns", 0))
            rbi = int(stats.get("rbi", 0))
        except:
            pass

    gut = random.uniform(0.5, 0.9)
    credits = random.uniform(8.0, 10.5)

    return {
        "player": player_name,
        "form_score": form_avg,
        "last_5_matches": last_5,
        "risk_score": risk,
        "vs_team_score": avg * 100,
        "vs_opponent_score": obp * 100,
        "gut_score": gut,
        "credits": credits,
        "role": role,
        "opponent_team": opponent,
        "homeRuns": hr,
        "rbi": rbi,
        "avg": avg,
        "obp": obp
    }

def build_feature_set(players, match):
    stats = [fetch_player_stats(p, match) for p in players]
    df = pd.DataFrame(stats)
    df["weather_multiplier"] = get_weather_impact(match)
    df["difficulty_multiplier"] = df["opponent_team"].apply(get_opponent_difficulty)
    return df

# =============================
# Predict Fantasy Points
# =============================
def predict_fantasy_points(df):
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    X = df[["form_score", "vs_team_score", "vs_opponent_score", "gut_score"]]
    y = (
        0.35 * df["form_score"] +
        0.25 * df["vs_team_score"] +
        0.2 * df["vs_opponent_score"] +
        0.2 * df["gut_score"] * 100
    )
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    preds = model.predict(X_scaled)
    residuals = (y - preds) ** 2
    confidence = np.exp(-residuals / np.mean(residuals))

    df["fps"] = preds * df["weather_multiplier"] * df["difficulty_multiplier"]
    df["confidence"] = confidence
    df["tag"] = ["✅ Safe" if c > 0.8 and r < 15 else "⚠️ Risky" for c, r in zip(df["confidence"], df["risk_score"])]
    df["impact_rating"] = 0.4 * df["fps"] + 0.4 * df["confidence"] * 100 - 0.2 * df["risk_score"]
    return df.sort_values(by="fps", ascending=False).reset_index(drop=True)

# =============================
# Team Generator
# =============================
def generate_team(df):
    team, credits = [], 0
    role_counts = {"Batter": 0, "Pitcher": 0, "Outfielder": 0}
    min_roles = {"Batter": 2, "Pitcher": 1, "Outfielder": 1}
    max_roles = {"Batter": 6, "Pitcher": 5, "Outfielder": 5}

    for _, row in df.sample(frac=1).iterrows():
        if (
            credits + row["credits"] <= 100 and
            len(team) < 11 and
            role_counts[row["role"]] < max_roles[row["role"]]
        ):
            team.append(row)
            credits += row["credits"]
            role_counts[row["role"]] += 1

    if len(team) == 11 and all(role_counts[r] >= min_roles[r] for r in min_roles):
        final = pd.DataFrame(team).reset_index(drop=True)
        final.at[0, "player"] += " (C)"
        final.at[1, "player"] += " (VC)"
        final.at[0, "fps"] *= 2
        final.at[1, "fps"] *= 1.5
        return final
    return pd.DataFrame()

def generate_multiple_teams(df, num_teams=3):
    teams = []
    for _ in range(num_teams):
        team = generate_team(df)
        if not team.empty:
            teams.append(team)
    return teams

# =============================
# Streamlit App UI
# =============================
st.title("⚾ Dream11 Baseball AI Team Builder")

match = st.selectbox("Choose an upcoming match:", get_upcoming_matches())
lineup = get_confirmed_lineup(match)

if not lineup:
    st.warning("Lineup not yet confirmed.")
else:
    df = build_feature_set(lineup, match)
    df = predict_fantasy_points(df)

    st.subheader("📊 Top Players")
    st.dataframe(df[["player", "fps", "confidence", "tag", "avg", "obp", "homeRuns", "rbi"]], use_container_width=True)

    st.subheader("🛠 Generated Teams")
    teams = generate_multiple_teams(df, num_teams=3)
    for i, team in enumerate(teams, 1):
        st.markdown(f"**Team #{i}**")
        st.dataframe(team[["player", "role", "fps", "confidence", "tag"]], use_container_width=True)
