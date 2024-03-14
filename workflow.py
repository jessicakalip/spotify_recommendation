# url = "https://wagon-public-datasets.s3.amazonaws.com/Machine%20Learning%20Datasets/ML_spotify_data.csv"
# df = pd.read_csv(url)


# # Define X and y
# X = df.drop(columns=["name", "artists"])  # Remove non numerical features
# y = df["tempo"]

# # Scale the features
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X)
# knn_model = KNeighborsRegressor().fit(X_scaled, y)  # Instanciate and train model


# mypip = Pipeline
# MinMaxScaler
# KNeighborsRegressor

# pickle


#  # Preprocessor
# num_transformer = make_pipeline(SimpleImputer(), StandardScaler())
# cat_transformer = OneHotEncoder()
# preproc = make_column_transformer(
#     (num_transformer, make_column_selector(dtype_include=['float6
# 4'])),
#     (cat_transformer, make_column_selector(dtype_include=['objec
# t','bool'])),
#     remainder='passthrough'
# )
# # Add estimator
# pipeline = make_pipeline(preproc, Ridge())
# pipeline
