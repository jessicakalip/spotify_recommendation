import pandas as pd
import streamlit as st
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
import string
from difflib import SequenceMatcher


# Cleans up the names and input (code from class)
def basic_cleaning(sentence):
    sentence = sentence.lower()
    sentence = "".join(char for char in sentence if not char.isdigit())
    sentence = "".join(char for char in sentence if char not in string.punctuation)
    return sentence.strip()


# Get the more similar songs
def get_most_similar(df, user_input, num_values=5):
    name_list = []
    cleaned_user_input = basic_cleaning(user_input)
    song_df = df.loc[:, ["name", "artists"]]
    song_df["name_artists"] = song_df["name"] + " " + song_df["artists"]
    song_df["name_artists"] = song_df["name_artists"].apply(basic_cleaning)
    song_df["sim_rating"] = song_df["name_artists"].apply(
        lambda x: SequenceMatcher(None, x, cleaned_user_input).ratio()
    )
    song_df = song_df.sort_values(by="sim_rating", ascending=False).reset_index(
        drop=True
    )
    name_list = song_df.loc[: num_values - 1, "name"].tolist()
    return name_list


url = "https://wagon-public-datasets.s3.amazonaws.com/Machine%20Learning%20Datasets/ML_spotify_data.csv"
df = pd.read_csv(url)


# Define X and y
X = df.drop(columns=["name", "artists"])  # Remove non numerical features
y = df["tempo"]

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
knn_model = KNeighborsRegressor().fit(X_scaled, y)  # Instanciate and train model


def create_playlist(input):
    input_song = df[df["name"] == input]

    X_new = input_song.drop(columns=["name", "artists"])
    X_new_scaled = scaler.transform(X_new)

    # Pass song to model, ask for 11 closest points, and unpack the corresponding indices to a list
    ind_list = list(knn_model.kneighbors(X_new_scaled, n_neighbors=11)[1][0])

    # Filter original dataframe with indices list and sort by tempo
    playlist = df.iloc[ind_list, :].sort_values(by="tempo")

    return playlist


page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: radial-gradient( circle farthest-corner at 10% 20%, rgba(222,168,248,1) 0%, rgba(168,222,258,1) 21.9%, rgba(189,250,205,1) 35.6%, rgba(243,250,189,1) 53.9%, rgba(250,227,189,1) 66.8%, rgba(248,172,171,1) 95%, rgba(254,170,212,1) 99.9% );
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown(
    """
<style>
.css-1v0mbdj > img{
    border-radius: 50%;
}
</style>
""",
    unsafe_allow_html=True,
)
left_co, cent_co, last_co = st.columns(3)
with cent_co:
    st.image("./pandamix.png", width=200)

title_text = "DJ Panda is here to save your night!! 🐼👯‍♀️🪩🎉🍾🎊🥳🍺🍷🍸🦄🌈🥰🤡🤖👾"
st.title(title_text)

st.write(
    "Music recommendation app based on the KNN model and trained on the Spotify Dataset."
)
input = st.text_input("Enter your favourite song  🎹🎸")
print(input)

suggested_songs = get_most_similar(df, input, num_values=5)
print(suggested_songs)


option = st.selectbox(
    "Songs related to your input, existing in our dataset", suggested_songs
)

st.write("You selected:", option)


# Create a button
if st.button("Generate Playlist"):
    # Call the function when the button is clicked
    result = create_playlist(option)

    result["artists_clean"] = result["artists"].map(lambda x: x[2:-2])
    result["name_artist"] = result["name"] + " - " + result["artists_clean"]
    songs = result["name_artist"]

    for idx, song in enumerate(songs):
        concatenated_song = song.replace(" ", "+")
        youtube_link = (
            f"https://www.youtube.com/results?search_query={concatenated_song}"
        )
        st.write(f"{idx+1}. {song} [Click to Listen](%s)" % youtube_link)
