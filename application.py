from flask import Flask,render_template, request
from recommender import recommend_random
from utils import movies
app = Flask(__name__)

@app.route('/')
def hello():
    """
    Returns:
        hello is printed out
    """
    return render_template('index.html',name ='Binomial Baharat',movies = movies.title.to_list())

@app.route('/recommend')
def recommendations():
    if request.args['algo']=='Random':
        recs = recommend_random() 
        print(request.args)

        titles = request.args.getlist('title')
        ratings = request.args.getlist('Ratings')
        user_input = dict(zip(titles,ratings))

        print(user_input)


        for keys in user_input:
            user_input[keys] = int(user_input[keys])


        return render_template('recommend.html',recs =recs)
    else:
        return f"Function not defined"

if __name__=='__main__':
    app.run(debug=True,port=5000)
