import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import random
import io

# =============================
# Upcoming Matches & Lineups
# =============================
def get_upcoming_matches():
    return ["Dodgers vs Cubs", "Giants vs Braves"]

def get_confirmed_lineup(match):
    try:
        url = "https://www.mlb.com/starting-lineups"
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        containers = soup.find_all("div", class_="lineup-game")

        selected_teams = match.split(" vs ")
        confirmed_players = []

        for game in containers:
            teams = game.find_all("span", class_="team-name")
            if len(teams) < 2:
                continue
            team1 = teams[0].text.strip()
            team2 = teams[1].text.strip()

            if selected_teams[0] in [team1, team2] and selected_teams[1] in [team1, team2]:
                players = game.find_all("a", class_="player-name")
                confirmed_players = [p.text.strip() for p in players]
                break

        return confirmed_players
    except Exception as e:
        print(f"Error scraping lineups: {e}")
        return []

# =============================
# Player Stats & Features
# =============================
def fetch_player_stats(player_name):
    try:
        search_url = f"https://statsapi.mlb.com/api/v1/people/search?name={player_name}"
        response = requests.get(search_url)
        data = response.json()
        if data['totalSize'] > 0:
            player_id = data['people'][0]['id']
            stats_url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=season&group=hitting"
            stats_resp = requests.get(stats_url)
            stats_data = stats_resp.json()
            stats = stats_data['stats'][0]['splits'][0]['stat']

            avg = float(stats.get('avg', 0))
            hr = int(stats.get('homeRuns', 0))
            rbi = int(stats.get('rbi', 0))

            form_score = min(100, (avg * 300 + hr * 2 + rbi))
            vs_team_score = 70

            weekday = datetime.today().weekday()
            bias = [0.9, 0.85, 0.75, 0.8, 0.95, 1.05, 1.0]
            gut_base = avg * bias[weekday]
            gut_random = random.uniform(0.9, 1.1)
            gut_score = round(min(1.0, max(0.5, gut_base * gut_random)), 2)

            consistency = round((form_score * 0.6 + gut_score * 100 * 0.4), 1)

            return {
                "player": player_name,
                "form_score": form_score,
                "vs_team_score": vs_team_score,
                "gut_score": gut_score,
                "credits": 9.0,
                "consistency": consistency
            }
    except Exception as e:
        print(f"Error fetching data for {player_name}: {e}")

    return {
        "player": player_name,
        "form_score": 50,
        "vs_team_score": 50,
        "gut_score": 0.6,
        "credits": 8.5,
        "consistency": 60.0
    }

def build_feature_set(players):
    stats = [fetch_player_stats(p) for p in players]
    return pd.DataFrame(stats)

# =============================
# Predict Fantasy Points
# =============================
def predict_fantasy_points(df):
    df['fps'] = 0.4 * df['form_score'] + 0.3 * df['vs_team_score'] + 0.3 * df['gut_score'] * 100
    return df.sort_values(by='fps', ascending=False).reset_index(drop=True)

# =============================
# Generate Multiple Dream11 Teams
# =============================
def generate_teams(df, num_teams=3):
    teams = []
    df = df.sort_values(by='fps', ascending=False).reset_index(drop=True)
    top_pool = df.head(15)

    for i in range(num_teams):
        shuffled = top_pool.sample(frac=1, random_state=i).reset_index(drop=True)
        team, credits = [], 0
        for _, row in shuffled.iterrows():
            if credits + row['credits'] <= 100 and len(team) < 11:
                team.append(row)
                credits += row['credits']

        final = pd.DataFrame(team).reset_index(drop=True)
        final['captain_score'] = 0.5 * final['fps'] + 0.3 * final['form_score'] + 0.2 * final['gut_score'] * 100
        final = final.sort_values(by='captain_score', ascending=False).reset_index(drop=True)
        roles = ['Captain', 'Vice-Captain'] + ['Player'] * (len(final) - 2)
        final['Role'] = roles
        final['final_fps'] = final.apply(lambda row: row['fps'] * 2 if row['Role'] == 'Captain' else (row['fps'] * 1.5 if row['Role'] == 'Vice-Captain' else row['fps']), axis=1)
        final['Team#'] = f'Team {i+1}'
        teams.append(final)

    return pd.concat(teams).reset_index(drop=True)

# =============================
# Streamlit App UI
# =============================
st.title("🏏 Dream11 Baseball AI Generator")

match = st.selectbox("Choose Match", get_upcoming_matches())
num_teams = st.slider("How many teams do you want to generate?", 1, 10, 3)

if st.button("Build My Teams"):
    players = get_confirmed_lineup(match)
    if not players:
        st.warning("Lineup not announced yet or unavailable.")
    else:
        features = build_feature_set(players)
        predicted = predict_fantasy_points(features)

        st.subheader("🎛️ Apply Filters")
        min_fps = st.slider("Minimum FPS", 0, 150, 60)
        max_credits = st.slider("Maximum Credits", 5.0, 12.0, 10.0)

        filtered = predicted[(predicted['fps'] >= min_fps) & (predicted['credits'] <= max_credits)]
        st.write("Filtered Players:")
        st.dataframe(filtered[['player', 'fps', 'credits', 'consistency']])

        if len(filtered) < 11:
            st.warning("Not enough players after filtering to create a full team.")
        else:
            final_teams = generate_teams(filtered, num_teams=num_teams)
            st.success(f"Here are your {num_teams} Dream11 teams with smart Captain/Vice-Captain")
            st.dataframe(final_teams[['Team#', 'player', 'Role', 'credits', 'fps', 'consistency', 'final_fps']])

            csv = final_teams.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download All Teams as CSV",
                data=csv,
                file_name="dream11_baseball_teams.csv",
                mime="text/csv"
            )

        st.subheader("🔍 Compare Two Players")
        player_list = list(filtered['player'])
        if len(player_list) >= 2:
            p1 = st.selectbox("Player A", player_list, key="p1")
            p2 = st.selectbox("Player B", player_list, key="p2")

            df_comp = filtered[filtered['player'].isin([p1, p2])].set_index("player")
            st.write("Comparison:")
            st.dataframe(df_comp[['fps', 'form_score', 'gut_score', 'consistency', 'credits']])
