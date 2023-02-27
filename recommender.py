"""
Contains various recommondation implementations
all algorithms return a list of movieids
"""

import pandas as pd
import numpy as np
from utils import movies
import pickle
from scipy.sparse import csr_matrix



def recommend_random(k=3):
    return movies['title'].sample(k).to_list()

def recommend_with_NMF(query, k=3):
    """
    NMF Recommender
    INPUT
    - user_vector with shape (1, #number of movies)
    - user_item_matrix
    - trained NMF model
    OUTPUT
    - a list of movieIds
    """
    with open('./nmf_recommender.pkl', 'rb') as file:
        model = pickle.load(file)

    print(model.components_.shape)
    
    data = list(query.values())
    print(data)         # the ratings of the new user
    row_ind=[0]*len(data)          # we use just a single row 0 for this user
    col_ind=list(query.keys())  
    
    user_vec= csr_matrix((data, (row_ind, col_ind)), shape=(1, model.components_.shape[1]))
    
    scores = model.inverse_transform(model.transform(user_vec))

    scores=pd.Series(scores[0])


    scores[query.keys()]=0
    
    scores=scores.sort_values(ascending=False)
    
    recommendations=scores.head(3).index

    return recommendations
    
#movies.set_index('movieId').loc[recommendations]
    
    


def recommend_neighborhood(query, k=3):
    """
    Filters and recommends the top k movies for any given input query based on a trained nearest neighbors model. 
    Returns a list of k movie ids.
    """   
    """
    Filters and recommends the top k movies for any given input query based on a trained nearest neighbors model. 
    Returns a list of k movie ids.
    """
    # 1. candiate generation

    ratings = pd.read_csv('./data/ratings.csv')
    ratings.drop('timestamp', axis=1, inplace=True)
    ratings.rename(
    columns={'userId':'user_id','movieId':'movie_id'},
    inplace=True
    )
    rating_count = ratings.groupby('movie_id')[['rating']].count()

# filter for movies with more than 20 ratings and extract the index
    popular_movies = rating_count[rating_count['rating']>20].index

# filter the ratings matrix and only keep the popular movies
    df = ratings[ratings['movie_id'].isin(popular_movies)].copy()

    user_ids = df['user_id'].unique()
    user_id_map = {v:k for k,v in enumerate(user_ids)}
    df['user_id'] = df['user_id'].map(user_id_map)


    movie_ids = df['movie_id'].unique()
    movie_id_map = {v:k for k,v in enumerate(movie_ids)}
    df['movie_id'] = df['movie_id'].map(movie_id_map)

    print(len(df['movie_id'].unique()))



    with open('./distance_recommender.pkl', 'rb') as file:
        model = pickle.load(file)
    # construct a user vector
    user_vec = np.repeat(0, len(df['movie_id'].unique()))
    
    for k,v in query.items():
        user_vec[k] = v
   
    # 2. scoring


    
    # find n neighbors

    similarity_scores, neighbor_ids = model.kneighbors(
    [user_vec],
    n_neighbors=10,
    return_distance=True
)

# sklearn returns a list of predictions
# extract the first and only value of the list

    neighbors = pd.DataFrame(
        data = {'neighbor_id': neighbor_ids[0], 'similarity_score': similarity_scores[0]}
    )

    neighbors.sort_values(
        by='similarity_score',
        ascending=False,
        inplace=True,
        ignore_index=True
    )

    neighborhood = df[df['user_id'].isin(neighbors['neighbor_id'])]
    
    # calculate their average rating
    
    df_score = neighborhood.groupby('movie_id')[['rating']].sum()
    df_score.rename(columns={'rating': 'score'}, inplace=True)
    df_score.reset_index(inplace=True)
    # 3. ranking
    
    # filter out movies allready seen by the user

    df_score['score'] = df_score['score'].map(lambda x: 0 if x in query.keys() else x)

# sort the scores from high to low 
    df_score.sort_values(
        by='score',
        ascending=False,
        inplace=True,
        ignore_index=True
    )

    
    # return the top-k highst rated movie ids or titles

    top3_scores = df_score.head(3)
    
    return list(top3_scores['movie_id'])
    
    

    
