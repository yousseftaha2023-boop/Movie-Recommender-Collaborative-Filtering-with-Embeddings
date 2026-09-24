import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split , cross_val_score , cross_val_predict , GroupShuffleSplit

df_movies = pd.read_csv(r'C:\Users\user\Downloads\archive (3)\movies.csv')
df_ratings = pd.read_csv(r'C:\Users\user\Downloads\archive (3)\ratings.csv')
# movies description
print(df_movies.head())
print(df_movies.shape)
print(df_movies.info())
print(df_movies.describe())
print(f'missing values {df_movies.isnull().sum()/len(df_movies)}')
# ratings description
print(df_ratings.head())
print(df_ratings.shape)
print(df_ratings.info())
print(df_ratings.describe())
print(f'missing values {df_ratings.isnull().sum()/len(df_ratings)}')
# df_ratings['mins_elapsed'] = (df_ratings['timestamp'] - df_ratings['timestamp'].min()) / 60
# df_ratings.drop(columns=['timestamp'], inplace=True)
print(df_ratings.head())
# train_test_split while enumerating the groups (user_id)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx , test_idx = next(gss.split(df_ratings, groups=df_ratings['userId']))
train_ratings = df_ratings.iloc[train_idx]
test_ratings = df_ratings.iloc[test_idx]

#collabrative filtering
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam

user_ids = df_ratings['userId'].unique()
movie_ids = df_ratings['movieId'].unique()

user2idx = {x: i for i, x in enumerate(user_ids)}
movie2idx = {x: i for i, x in enumerate(movie_ids)}

train_ratings['user_idx'] = train_ratings['userId'].map(user2idx)
train_ratings['movie_idx'] = train_ratings['movieId'].map(movie2idx)

test_ratings['user_idx'] = test_ratings['userId'].map(user2idx)
test_ratings['movie_idx'] = test_ratings['movieId'].map(movie2idx)


global_mean = train_ratings['rating'].mean()
movie_means = train_ratings.groupby('movie_id')['rating'].mean().to_dict()



train_ratings['movie_mean'] = train_ratings['movie_id'].map(movie_means)
train_ratings['rating_norm'] = train_ratings['rating'] - train_ratings['movie_mean']

test_ratings['movie_mean'] = test_ratings['movie_id'].map(movie_means)
test_ratings['rating_norm'] = test_ratings['rating'] - test_ratings['movie_mean']

num_users = len(user2idx)
num_movies = len(movie2idx)

embedding_size = 32  #latent factors size


user_input = tf.keras.layers.Input(shape=(1,), name='user_input')
movie_input = tf.keras.layers.Input(shape=(1,), name='movie_input')

user_embedding = tf.keras.layers.Embedding(num_users, embedding_size, name='user_embedding')(user_input)
movie_embedding = tf.keras.layers.Embedding(num_movies, embedding_size, name='movie_embedding')(movie_input)


user_vec = tf.keras.layers.Flatten()(user_embedding)
movie_vec = tf.keras.layers.Flatten()(movie_embedding)

dot_product = tf.keras.layers.Dot(axes=1)([user_vec, movie_vec])

user_bias = tf.keras.layers.Flatten()(tf.keras.layers.Embedding(num_users, 1, name='user_bias')(user_input))



output = tf.keras.layers.Add()([dot_product, user_bias])

# Model Definition
model = tf.keras.Model(inputs=[user_input, movie_input],  outputs=output)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=['mae']
)

# Train data - use mapped index columns
X_train_user = train_ratings['user_idx'].values
X_train_movie = train_ratings['movie_idx'].values
y_train = train_ratings['rating_norm'].values

# Test data - use mapped index columns
X_test_user = test_ratings['user_idx'].values
X_test_movie = test_ratings['movie_idx'].values
y_test = test_ratings['rating_norm'].values


history = model.fit(
    x=[X_train_user, X_train_movie],
    y=y_train,
    batch_size=256,
    epochs=10)



print(f'model summary' ,model.summary())
print(f'model evaluation', model.evaluate(x=[X_test_user, X_test_movie], y=y_test))



user_id = int(input("Enter User ID: "))
movie_id = int(input("Enter Movie ID: "))


u_idx = user2idx[user_id]
m_idx = movie2idx[movie_id]

pred_norm = model.predict([np.array([u_idx]), np.array([m_idx])], verbose=0)[0][0]
m_mean = movie_means.get(movie_id, global_mean)
rating = pred_norm + m_mean # add global mean to predicted rating value explicitly
print(f"Predicted Rating: {rating:.2f}")

