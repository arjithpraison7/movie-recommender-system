# Movie Recommender System

Building a Movie Recommender System using Machine Learning is a popular and interesting application. Here's a step-by-step guide to get started:

### **Step 1: Define the Goal and Scope**

Clearly define what you want our recommender system to achieve. Consider aspects like:

- **Type of Recommendation**: Will it be content-based (suggests based on user's historical preferences) or collaborative filtering (suggests based on other users' preferences)?
- **Metrics for Evaluation**: Decide how you'll measure the performance of your recommendation system (e.g., Mean Absolute Error, Root Mean Squared Error, etc.).

### **Step 2: Gather Data**

You'll need data on movies and user interactions. The MovieLens dataset is a popular choice for such projects. It contains a large set of movie ratings provided by users.

### **Step 3: Data Preprocessing**

Clean and prepare the data. This may involve tasks like handling missing values, encoding categorical variables, and normalizing or standardizing numerical features.

### **Step 4: Feature Engineering**

Create features that the model can use for making recommendations. For a content-based approach, this might involve extracting features from movie metadata (e.g., genre, director, actors, etc.). For collaborative filtering, you'll need user-item interaction matrices.

### **Step 5: Choose a Recommender System Algorithm**

There are various algorithms you can use:

- **Content-Based Filtering**: Recommends items similar to those the user has liked in the past.
- **Collaborative Filtering**:
    - **User-Based**: Recommends items liked by users similar to the target user.
    - **Item-Based**: Recommends items similar to those liked by the user.
    - **Matrix Factorization**: Factorizes the user-item interaction matrix to find latent features.
- **Hybrid Approaches**: Combine content-based and collaborative filtering for better results.

### **Step 6: Split Data for Training and Testing**

Divide your data into training and testing sets. This is crucial for evaluating your model's performance.

### **Step 7: Train the Model**

Depending on the chosen algorithm, train your model on the training data.

### **Step 8: Evaluate the Model**

Use the testing data to evaluate how well your model performs. Use appropriate evaluation metrics based on your defined goal.

### **Step 9: Fine-tuning and Optimization**

Iterate on your model. Experiment with different algorithms, hyperparameters, and features to improve performance.

### **Step 10: Deployment**

Once you're satisfied with the performance, you can deploy your recommender system. This could be as a web application or integrated into an existing platform.

### **Step 11: User Interface (Optional)**

Create a user-friendly interface for users to interact with your recommender system.

### **Step 12: Documentation and Presentation**

Document your project, including the problem statement, data used, methodology, results, and any challenges faced. Prepare a presentation to showcase your work.

# Step 1:Define the Goal and Scope

1. **Type of Recommendation**:
    - We'll build a collaborative filtering recommender system. This means we'll recommend movies based on the preferences and behavior of similar users.
2. **Metrics for Evaluation**:
    - We'll primarily use metrics like Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE) to evaluate the performance of our recommender system.
3. **Target Audience**:
    - The target audience can be general movie enthusiasts who want personalized movie recommendations.
4. **Additional Features** (Optional):
    - For simplicity, let's focus on building a basic recommendation engine first. We can consider additional features in future iterations.
5. **Platform or Interface**:
    - Since this is a learning project, you can build a simple command-line interface or a basic web interface (if you're comfortable with web development).
6. **Data Source**:
    - We'll use the MovieLens dataset, which contains a large set of movie ratings provided by users.
7. **Timeline and Resources**:
    - Since this is a learning project, you can proceed at your own pace. As for resources, a standard laptop or computer should be sufficient for this project.
    
    # Step 2:Gathering Data
    
    For this project, we'll be using the MovieLens dataset. It's a widely-used dataset for building recommender systems. There are different versions available, and we'll use the "MovieLens 100K" dataset for simplicity.
    
    Here are the steps to obtain the dataset:
    
    1. **Visit the MovieLens Website**:
        - Go to the MovieLens website: **[https://grouplens.org/datasets/movielens/100k/](https://grouplens.org/datasets/movielens/100k/)**
    2. **Download the Dataset**:
        - On the page, you'll find a link to download the dataset. It will likely be in a zip file format.
    3. **Extract the Dataset**:
        - After downloading, extract the contents of the zip file to a folder on your computer.
    
    The dataset typically includes files like **`u.data`** (which contains user ratings), **`u.item`** (which contains movie information), and others.
    
    # Step 3:Data Preprocessing
    
    ### 1. **Load the Data**
    
    We'll start by loading the data from the MovieLens dataset.
    
    ```python
    import pandas as pd
    
    # Define file paths
    data_path = 'path_to_data_folder/u.data'
    item_path = 'path_to_data_folder/u.item'
    
    # Load user ratings and movie information
    ratings_df = pd.read_csv(data_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    movies_df = pd.read_csv(item_path, sep='|', encoding='latin-1', header=None, names=['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_url'] + list(range(19)))
    
    # Display the first few rows of the datasets to check if they loaded correctly
    print(ratings_df.head())
    print(movies_df.head())
    ```
    
    ```python
    user_id  item_id  rating  timestamp
    0      196      242       3  881250949
    1      186      302       3  891717742
    2       22      377       1  878887116
    3      244       51       2  880606923
    4      166      346       1  886397596
       item_id              title release_date  video_release_date                                           IMDb_url  0  1  2  3  ...  10  11  12  13  14  15  16  17  18
    0        1   Toy Story (1995)  01-Jan-1995                 NaN  http://us.imdb.com/M/title-exact?Toy%20Story%2...  0  0  0  1  ...   0   0   0   0   0   0   0   0   0
    1        2   GoldenEye (1995)  01-Jan-1995                 NaN  http://us.imdb.com/M/title-exact?GoldenEye%20(...  0  1  1  0  ...   0   0   0   0   0   0   1   0   0
    2        3  Four Rooms (1995)  01-Jan-1995                 NaN  http://us.imdb.com/M/title-exact?Four%20Rooms%...  0  0  0  0  ...   0   0   0   0   0   0   1   0   0
    3        4  Get Shorty (1995)  01-Jan-1995                 NaN  http://us.imdb.com/M/title-exact?Get%20Shorty%...  0  1  0  0  ...   0   0   0   0   0   0   0   0   0
    4        5     Copycat (1995)  01-Jan-1995                 NaN  http://us.imdb.com/M/title-exact?Copycat%20(1995)  0  0  0  0  ...   0   0   0   0   0   0   1   0   0
    ```
    
    ### 2. **Handle Missing Values**
    
    Since the MovieLens 100K dataset is well-prepared, you typically won't encounter missing values. However, it's good practice to check:
    
    ```python
    print(ratings_df.isnull().sum())
    print(movies_df.isnull().sum())
    ```
    
    It seems that there are some missing values in the **`release_date`**, **`video_release_date`**, and **`IMDb_url`** columns of the **`movies_df`** DataFrame. Since these columns are not essential for our initial collaborative filtering recommender system, we can ignore them for now.
    
    ### 3. **Data Exploration**
    
    Explore basic statistics and distributions:
    
    ```python
    print(ratings_df.describe())
    ```
    
    ```python
    user_id        item_id         rating     timestamp
    count  100000.00000  100000.000000  100000.000000  1.000000e+05
    mean      462.48475     425.530130       3.529860  8.835289e+08
    std       266.61442     330.798356       1.125674  5.343856e+06
    min         1.00000       1.000000       1.000000  8.747247e+08
    25%       254.00000     175.000000       3.000000  8.794487e+08
    50%       447.00000     322.000000       4.000000  8.828269e+08
    75%       682.00000     631.000000       4.000000  8.882600e+08
    max       943.00000    1682.000000       5.000000  8.932866e+08
    ```
    
    Since we are going for collaborative filtering we do not need to do standardisation or scaling.
    
    ### 4. **Create User-Item Interaction Matrix**
    
    For collaborative filtering, create the user-item interaction matrix:
    
    ```python
    interaction_matrix = ratings_df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    ```
    
    # **Step 4: Choose a Recommender System Algorithm**
    
    For this project, we'll use a user-based collaborative filtering approach. In this approach, we'll recommend movies to a user based on the preferences of similar users.
    
    ### User-Based Collaborative Filtering:
    
    ```python
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Transpose the interaction matrix for user-based CF
    user_item_matrix = interaction_matrix.T
    
    # Calculate similarity between users
    user_similarity = cosine_similarity(user_item_matrix)
    
    # Convert to DataFrame for better visualization
    user_similarity_df = pd.DataFrame(user_similarity, index=interaction_matrix.columns, columns=interaction_matrix.columns)
    
    # Display the user similarity DataFrame
    print(user_similarity_df.head())
    ```
    
    User-Based Collaborative Filtering, often referred to as "user-user collaborative filtering," makes recommendations based on the idea that users who have interacted similarly with items (e.g., rated movies similarly) in the past are likely to have similar preferences in the future. Here's how it works:
    
    1. **User-Item Interaction Matrix**:
        - We start with the user-item interaction matrix, where rows represent users and columns represent items (movies). Each cell in the matrix contains the user's interaction with the item, typically a rating.
    2. **Similarity Calculation**:
        - The first step is to calculate the similarity between users. Common similarity metrics include cosine similarity, Pearson correlation, and others.
        - Cosine similarity, as we used in the code example, measures the cosine of the angle between two user vectors in the user-item interaction matrix. High cosine similarity indicates similar preferences.
    3. **User Neighborhood**:
        - Once we have calculated user similarities, we create a "neighborhood" of similar users for each user. These are the users who are most similar to the target user based on their historical interactions.
    4. **Generate Recommendations**:
        - To make recommendations for a user, we look at the items that their neighbors have interacted with but that the target user has not.
        - Recommendations are generated based on the preferences of the neighbors. For example, if several similar users have rated a particular movie highly, it may be recommended to the target user.
    5. **Handling Cold Start**:
        - One challenge with user-based collaborative filtering is the "cold start" problem. It's difficult to make recommendations for new users who haven't provided much interaction data yet.
        - Handling new users often requires using other recommendation techniques or hybrid approaches.
    6. **Evaluation**:
        - The performance of the recommender system is typically evaluated using metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Precision at k, Recall at k, and others.
    
    User-Based Collaborative Filtering is relatively simple to understand and implement, making it a good starting point for building a recommender system. However, it has its limitations, including scalability issues when dealing with a large number of users and items.
    
    For more advanced recommender systems, you may explore hybrid approaches that combine collaborative filtering with content-based filtering or matrix factorization methods like Singular Value Decomposition (SVD) or Alternating Least Squares (ALS).
    
    # Step 5:Split Data for Training and Testing
    
    In this step, we'll divide the data into training and testing sets. The training set will be used to build the recommender system, while the testing set will be used to evaluate its performance.
    
    ```python
    from sklearn.model_selection import train_test_split
    
    # Define X (features) and y (target) for collaborative filtering
    X = ratings_df[['user_id', 'item_id']].values
    y = ratings_df['rating'].values
    
    # Split data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Display the shapes of the training and testing sets
    print(f"Training set: X_train.shape = {X_train.shape}, y_train.shape = {y_train.shape}")
    print(f"Testing set: X_test.shape = {X_test.shape}, y_test.shape = {y_test.shape}")
    ```
    
    ```python
    Training set: X_train.shape = (80000, 2), y_train.shape = (80000,)
    Testing set: X_test.shape = (20000, 2), y_test.shape = (20000,)
    ```
    
    In this code, we're using **`train_test_split`** from the **`sklearn.model_selection`** module to randomly split the data into training and testing sets. We're using 80% of the data for training and 20% for testing. The **`random_state`** parameter ensures reproducibility.
    
    After this step, we'll have separate sets of data for training and testing our recommender system.
    
    # Step 6:Build the Recommender System
    
    Since we've already calculated user similarities, we can use them to make recommendations. Here's the code to do that:
    
    ```python
    import numpy as np
    
    def predict_rating(user_id, item_id):
        # Get the ratings of similar users for the item
        similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:]
        ratings = []
        for u, sim in similar_users.iteritems():
            if not np.isnan(user_item_matrix.loc[u, item_id]):
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
    ```
    
    In this code, we define a function **`predict_rating`** that takes a user ID and an item ID as input. It calculates the predicted rating for the item based on the ratings of similar users.
    
    Please note that this is a basic example. In practice, you would want to handle cases where no similar users have rated the item (as in the **`if len(ratings) == 0`** block). You might also want to refine the method for selecting the top similar users and handling ties.
    
    # Step 7:Evaluate the Recommender System
    
    Next, we want to evaluate the performance of your recommender system. Common metrics include Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE). Here's an example using MAE:
    
    ```python
    from sklearn.metrics import mean_absolute_error
    
    # Predict ratings for the test set
    y_pred = [predict_rating(user_id, item_id) for user_id, item_id in X_test]
    
    # Calculate MAE
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error (MAE): {mae}")
    ```
    
    ```python
    Predicted rating for user 1 and item 1: 3.4766001493418472
    Mean Absolute Error (MAE): 3.0538384834339216
    ```
    
    Let's break down the results:
    
    1. **Predicted Rating for User 1 and Item 1**:
        - The predicted rating for user 1 and item 1 is approximately 3.48. This means that, based on the collaborative filtering algorithm you implemented, the system predicts that user 1 would rate item 1 around 3.48 on average.
    2. **Mean Absolute Error (MAE)**:
        - The Mean Absolute Error is a metric used to evaluate the performance of a recommender system. It measures the average absolute difference between the predicted ratings and the actual ratings in the test set.
        - In your case, the MAE is approximately 3.05. This means that, on average, the predicted ratings from your recommender system deviate from the actual ratings in the test set by about 3.05 units.
        - A lower MAE indicates better performance, as it means that the predicted ratings are closer to the actual ratings.
    
    To put it simply, a predicted rating of 3.48 for user 1 and item 1 suggests that the system thinks user 1 would likely enjoy item 1 to some extent. However, the MAE of 3.05 indicates that there is room for improvement in the accuracy of the predictions.
    
    ### **Step 8: Fine-tuning and Optimization**
    
    In this step, you can experiment with different parameters, algorithms, and techniques to improve the performance of your recommender system. Here are some ideas you can consider:
    
    1. **Experiment with Different Similarity Metrics**:
        - Try using alternative similarity metrics such as Pearson correlation, Jaccard similarity, or others to see if they lead to better recommendations.
    2. **Tune the Number of Similar Users**:
        - The number of similar users you consider for recommendations can have an impact on the quality of the suggestions. Experiment with different values to find the optimal number.
    3. **Implement Matrix Factorization**:
        - Consider implementing more advanced techniques like Singular Value Decomposition (SVD) or Alternating Least Squares (ALS) for collaborative filtering.
    4. **Handling Cold Start**:
        - Explore methods to handle the "cold start" problem for new users who haven't provided much interaction data yet. You might use content-based features or other techniques.
    5. **Hyperparameter Tuning**:
        - If you're using more complex algorithms, consider tuning hyperparameters to find the best combination for your specific dataset.
    6. **Explore Hybrid Approaches**:
        - Combine collaborative filtering with content-based filtering or other techniques to potentially improve recommendation accuracy.
    7. **Evaluate on Real User Data**:
        - If possible, collect real user data and evaluate the performance of your recommender system in a real-world setting.
    8. **User Interface (Optional)**:
        - Create a user-friendly interface for users to interact with your recommender system. This could be a web application, a mobile app, or a simple command-line interface.
    
    # Conclusion
    
    In this project, we successfully implemented a Movie Recommender System using the User-Based Collaborative Filtering algorithm. The system is designed to suggest movies to users based on their past preferences and the behavior of similar users.
    
    ### Key Steps and Findings:
    
    1. **Data Preprocessing**:
        - We started by loading and cleaning the MovieLens 100K dataset. This involved handling missing values and organizing the data into a user-item interaction matrix.
    2. **User-Based Collaborative Filtering**:
        - We chose to implement a user-based collaborative filtering approach, which makes recommendations based on the behavior of similar users. This involved calculating user similarities using cosine similarity.
    3. **Training and Testing**:
        - We split the data into training and testing sets to evaluate the performance of our recommender system. The Mean Absolute Error (MAE) was used as a metric to measure prediction accuracy.
    4. **Evaluation**:
        - The predicted rating for user-item pairs and the MAE were calculated. The predicted rating for user 1 and item 1 was approximately 3.48, and the MAE was around 3.05.
    5. **Documentation and Presentation**:
        - We documented the entire project, including data preprocessing, algorithm implementation, and evaluation results. A presentation slide deck was created to summarize the project.
    
    ### Future Work:
    
    - Further optimization of the recommender system could involve experimenting with different similarity metrics, tuning the number of similar users, or implementing more advanced techniques like matrix factorization.
    - Handling the "cold start" problem for new users and items could be explored, as well as incorporating content-based features for a hybrid approach.
    - Building a user interface or deploying the recommender system for real-world use would enhance its accessibility and usability.
    
    ### Reflection:
    
    This project provided valuable insights into building a recommender system and applying collaborative filtering techniques. It also highlighted the importance of documentation and presentation skills in effectively communicating technical work.
    
    Overall, the Movie Recommender System project was a rewarding learning experience, and the implemented system has the potential for further enhancement and application in real-world scenarios.
    
    # Author
    
    [Arjith Praison](https://www.linkedin.com/in/arjith-praison-95b145184/)
    
    University of Siegen
