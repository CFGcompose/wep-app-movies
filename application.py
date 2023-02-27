from flask import Flask,render_template, request
from recommender import recommend_random
from recommender import recommend_with_NMF
from recommender import recommend_neighborhood

from utils import movies, movie_to_id, id_to_movie




app = Flask(__name__)

@app.route('/')
def hello():
    """
    Returns:
        hello is printed out
    """
    return render_template('index.html',name ='Christoph',movies = movies.title.to_list())

@app.route('/recommend')
def recommendations():
    if request.args['algo']=='Random':
        recs = recommend_random() 
        print(request.args)
        return render_template('recommend.html',recs =recs)

    elif request.args['algo']=='NMF':
        
        #print(request.args)

        titles = request.args.getlist('title')
        ratings = request.args.getlist('Ratings')
        user_input = dict(zip(titles,ratings))

        print(titles)
        #print(ratings)
        #print(user_input)
        input_id_list =[]
        for title in titles:
            input_ids = movie_to_id(title)
            input_id_list.append(int(input_ids))

        rating_list = []
        for rating in ratings:
            rating_list.append(int(rating))



        print(input_id_list)

        query = dict(zip(input_id_list,rating_list))

        print(query.keys(), query.values())

        recs = recommend_with_NMF(query, k=3)

        print(recs)

        recs = id_to_movie(recs)


        return render_template('recommend_NMF.html',recs =recs)
    
    elif request.args['algo']=='Cosine similarity':
        
        titles = request.args.getlist('title')
        ratings = request.args.getlist('Ratings')
        user_input = dict(zip(titles,ratings))

        print(titles)
        #print(ratings)
        #print(user_input)
        input_id_list =[]
        for title in titles:
            input_ids = movie_to_id(title)
            input_id_list.append(int(input_ids))

        rating_list = []
        for rating in ratings:
            rating_list.append(int(rating))



        print(input_id_list)

        query = dict(zip(input_id_list,rating_list))
    

        recs = recommend_neighborhood(query, k=3)

        print(recs)

        recs = id_to_movie(recs)


        return render_template('recommend_sim.html',recs =recs)

    else:
        return f"Function not defined"

if __name__=='__main__':
    app.run(debug=True,port=5000)
