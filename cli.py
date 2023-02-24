from recommender import recommend_random
from utils import movies

### Terminal recommender:
if __name__=='__main__': ## Runs the app (main module) 
    print('>>>> Here are some movie recommendations for you<<<<')
    print('')
    print('Random movies')
    movie_ids = recommend_random(3)
    print(movie_ids)


