"""
Contains various recommondation implementations
all algorithms return a list of movieids
"""

import pandas as pd
import numpy as np
from utils import movies

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
    pass

def recommend_neighborhood(query, model, k=3):
    """
    Filters and recommends the top k movies for any given input query based on a trained nearest neighbors model. 
    Returns a list of k movie ids.
    """   
    pass
    
