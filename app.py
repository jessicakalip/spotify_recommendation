import pandas as pd
import streamlit as st
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler

"""
TO DO:
- Make case-insensitive input / have dropdown with similar song names / suggestions 
- Put the code onto Streamlit Cloud
- Have a song player button next to each song in the dataframe? 
- Spotify API with more songs
"""


## ALL FUNCTIONAL CODE
def create_playlist(input):
    url = "https://wagon-public-datasets.s3.amazonaws.com/Machine%20Learning%20Datasets/ML_spotify_data.csv"
    df = pd.read_csv(url)
    # queen_song = df.iloc[4295:4296]

    input_song = df[df["name"] == input]
    # Define X and y
    X = df.drop(columns=["name", "artists"])  # Remove non numerical features
    y = df["tempo"]

    # Scale the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    knn_model = KNeighborsRegressor().fit(X_scaled, y)  # Instanciate and train model

    X_new = input_song.drop(columns=["name", "artists"])
    X_new_scaled = scaler.transform(X_new)

    knn_model.kneighbors(
        X_new_scaled, n_neighbors=2
    )  # Return the distances and index of the 2 closest points

    # Pass song to model, ask for 11 closest points, and unpack the corresponding indices to a list
    ind_list = list(knn_model.kneighbors(X_new_scaled, n_neighbors=11)[1][0])

    # Filter original dataframe with indices list and sort by tempo
    playlist = df.iloc[ind_list, :].sort_values(by="tempo")

    return playlist


# data = st.button("Generate Playlist", on_click=create_playlist())


def main():
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
        st.image("./pandamix.png", width=300)

    input = st.text_input("Enter your favourite song  🎹🎸")
    print(input)

    # Create a button
    if st.button("Generate Playlist"):
        # Call the function when the button is clicked
        result = create_playlist(input)

        # Display the result
        st.dataframe(result)


if __name__ == "__main__":
    main()
