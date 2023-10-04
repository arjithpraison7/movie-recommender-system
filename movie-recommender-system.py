import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Define file paths
data_path = 'ml-100k/u.data'
item_path = 'ml-100k/u.item'

# Load user ratings and movie information
ratings_df = pd.read_csv(data_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
movies_df = pd.read_csv(item_path, sep='|', encoding='latin-1', header=None, names=['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_url'] + list(range(19)))

# Display the first few rows of the datasets to check if they loaded correctly
print(ratings_df.head())
print(movies_df.head())

# Handle missing values
print(ratings_df.isnull().sum())
print(movies_df.isnull().sum())

# Basic Data Exploration
print(ratings_df.describe())

# User Item interaction matrix
interaction_matrix = ratings_df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

# Transpose the interaction matrix for user-based CF
user_item_matrix = interaction_matrix.T

# Calculate similarity between users
user_similarity = cosine_similarity(user_item_matrix)

# Convert to DataFrame for better visualization
user_similarity_df = pd.DataFrame(user_similarity, index=interaction_matrix.columns, columns=interaction_matrix.columns)

# Display the user similarity DataFrame
print(user_similarity_df.head())

# Define X (features) and y (target) for collaborative filtering
X = ratings_df[['user_id', 'item_id']].values
y = ratings_df['rating'].values

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the training and testing sets
print(f"Training set: X_train.shape = {X_train.shape}, y_train.shape = {y_train.shape}")
print(f"Testing set: X_test.shape = {X_test.shape}, y_test.shape = {y_test.shape}")


def predict_rating(user_id, item_id):
    # Get the ratings of similar users for the item
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:]
    ratings = []
    for u, sim in similar_users.items():
        if item_id in user_item_matrix.columns and not np.isnan(user_item_matrix.loc[u, item_id]):
            ratings.append((sim, user_item_matrix.loc[u, item_id]))
        if len(ratings) >= 5:  # Get ratings from top 5 similar users
            break
    if len(ratings) == 0:
        return 0  # If no similar users have rated the item, return a default value (e.g., 0)
    else:
        return sum(sim * rating for sim, rating in ratings) / sum(sim for sim, _ in ratings)

# Example: Predict rating for user 1 and item 1
predicted_rating = predict_rating(1, 1)
print(f"Predicted rating for user 1 and item 1: {predicted_rating}")



# Predict ratings for the test set
y_pred = [predict_rating(user_id, item_id) for user_id, item_id in X_test]

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")
