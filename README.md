# Movie Recommender: Collaborative Filtering with Embeddings

A neural collaborative filtering model that predicts a user's rating for a movie, built with TensorFlow/Keras on the [MovieLens](https://grouplens.org/datasets/movielens/) `movies.csv` / `ratings.csv` dataset. The model learns latent user and movie embeddings and combines them with a user bias term, and lets you query a predicted rating for any user/movie pair interactively.

## Overview

1. Load and inspect the movies and ratings data
2. Group-aware train/test split, so a user's ratings don't leak across splits
3. Map raw user and movie IDs to contiguous embedding indices
4. Normalize ratings against each movie's mean (bias correction)
5. Train an embedding-based neural network to predict the normalized rating
6. Evaluate on the held-out set
7. Interactive prediction for a chosen user/movie pair

## Approach

### Data splitting

`GroupShuffleSplit` (`test_size=0.2`, `random_state=42`) splits on `userId`, so all of a given user's ratings land entirely in train or entirely in test. This avoids the leakage that a plain random split would introduce, since the model would otherwise see some of a user's preferences during training and be evaluated on the rest of the same user.

### ID mapping

Since `userId` and `movieId` values aren't contiguous, they're mapped to dense zero-based indices (`user2idx`, `movie2idx`) built from the full dataset, which the embedding layers use as lookup indices.

### Rating normalization

Each rating is centered on its movie's mean rating in the train set (`rating_norm = rating - movie_mean`). The network is trained to predict this residual rather than the raw rating, and the movie mean is added back at prediction time. This removes each movie's baseline popularity from the signal the embeddings need to learn.

### Model architecture

| Component | Configuration |
|-----------|---------------|
| User embedding | `num_users x 32` |
| Movie embedding | `num_movies x 32` |
| User bias | `num_users x 1` |
| Interaction | Dot product of user and movie embedding vectors |
| Output | Dot product + user bias |

Training settings:

- Loss: mean squared error; metric: MAE
- Optimizer: Adam, learning rate `0.001`
- 10 epochs, batch size 256
- Seed fixed with `tf.random.set_seed(42)`

### Prediction

After training, the script prompts for a user ID and a movie ID, looks up their embedding indices, predicts the normalized rating, and adds the movie's mean rating back to produce the final predicted rating.

## Getting started

### Prerequisites

- Python 3.9+

```bash
pip install numpy pandas scikit-learn tensorflow
```

### Data

Download the MovieLens dataset (e.g. the [ml-latest-small](https://grouplens.org/datasets/movielens/) release) and place `movies.csv` and `ratings.csv` where the script can find them. The data is not included in this repository.

> [!IMPORTANT]
> The script currently reads files using absolute Windows paths. Update the two `pd.read_csv(...)` calls near the top of `recommendation_system.py` to match your machine before running.

### Run

```bash
python recommendation_system.py
```

The script trains the model, prints a summary and evaluation metrics, then prompts:

```text
Enter User ID:
Enter Movie ID:
```

and prints the predicted rating for that pair.

## Project structure

```text
.
├── recommendation_system.py   # Data loading, preprocessing, embedding model, evaluation, interactive prediction
└── README.md
```

## Next steps

- Add movie-genre or user-demographic side features for a hybrid (content + collaborative) model
- Regularize the embeddings to reduce overfitting on sparse users/movies
- Precompute top-N recommendations per user instead of single on-demand predictions
- Track RMSE alongside MAE for easier comparison with published MovieLens benchmarks
