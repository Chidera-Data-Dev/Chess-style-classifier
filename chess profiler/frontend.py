import pandas as pd
import plotly.express as px
import streamlit as st
from backend import (load_model_artifacts, fetch_lichess_profile, classify_player, render_avatar, 
                     FEATURE_COLUMNS, render_placeholder_avatar, CLUSTER_AVATARS, 
                     CLUSTER_LABELS, CLUSTER_MESSAGES, DEFAULT_AVATAR)

# UI

st.set_page_config(page_title="Chess Player Style Classifier", page_icon="♟️")
st.title("♟️ Chess Player Style Classifier")
st.subheader(
"Analyze a Lichess username to generate a playing-style profile from recent rated rapid games."
)
st.divider()
username_input = st.text_input("Enter your Lichess username", placeholder="e.g. DrNykterstein")
submit = st.button("Analyze")

if submit and username_input.strip():
    username = username_input.strip()

    with st.spinner(f"Looking up {username}..."):
        profile = fetch_lichess_profile(username)

    if profile is None:
        st.error(f"Couldn't find a Lichess user named '{username}'. Check the spelling.")
    else:
        with st.spinner("Fetching recent games and computing style profile — this can take a minute..."):
                    scaler, kmeans = load_model_artifacts()
                    cluster_id, result = classify_player(username, scaler, kmeans)
        col1, col2 = st.columns([1, 3])

        with col1:
            if cluster_id is not None:
                render_avatar(cluster_id)
            else:
                render_placeholder_avatar()

        with col2:
            st.subheader(profile.get("username", username))
            rapid_rating = profile.get("perfs", {}).get("rapid", {}).get("rating")
            if rapid_rating:
                st.write(f"**Rapid rating:** {rapid_rating}")
            else:
                st.write("**Rapid rating:** No rapid rating on record")

        st.divider()

        if cluster_id is None:
            st.warning(result)
        else:
            label = CLUSTER_LABELS.get(cluster_id, f"Cluster {cluster_id}")
            message = CLUSTER_MESSAGES.get(cluster_id, "")

            st.success(f"**Playing style: {label}**")
            st.write(message)

            with st.expander("Explore the numbers behind your result"):
                st.write("Your interactive plot:")
                feature_display = pd.DataFrame([result])[FEATURE_COLUMNS]
                # manually scaling the larger column
                feature_display["avg_minor_pieces_developed"] = feature_display["avg_minor_pieces_developed"] / 4
                fig = px.bar(feature_display, barmode="group", title="Player Behaviour")
                fig.update_layout(xaxis_title="Features",yaxis_title="Value")
                fig.update_xaxes(showticklabels=False)
                st.plotly_chart(fig, use_container_width=True)
                st.write("Feature Explanation:")
                st.write("""
                * **pct_games_never_pushed_contested_pawn:** How often you advance pawns into contested central territory (past the board's midline) during a game.
                * **avg_castle_phase:** How early or late you castle relative to the length of the game.
                * **pct_games_never_castled:** How often you never castle at all during a game.
                * **pct_games_queen_never_moved:** How often your queen never leaves its starting square during a game.
                * **avg_minor_dev_phase:** How early, on average, you develop your minor pieces (bishops and knights).
                * **avg_minor_pieces_developed:** How thoroughly you develop your minor pieces, out of a possible 4 (two bishops, two knights).
                                                """)

elif submit:
    st.warning("Enter a username first.")
