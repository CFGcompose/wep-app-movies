from flask import Flask
from simple_recommender import get_recommendations
from flask import render_template
from flask import request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', title='Hello, World!')
    
@app.route('/recommender')
def recommender():
    html_form_data = dict(request.args)
    # a python dictionary consisting of
    # "name"-value pairs from the HTML form!

    recs = get_recommendations()
    # at this point, we would then pass this
    #information as an argument into our recommender function.

    print(html_form_data)

    return render_template('recommendations.html',
                            movies = recs)
    
if __name__ == "__main__":
    app.run(debug=True, port=5000)
