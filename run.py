import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/disasterdata.db')
df = pd.read_sql_table('disasterdata.db', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    Y = df.drop(['message','id','original','genre'], axis =1)
    categories_list = Y.sum()
    categories_names = list(categories_list.index)
    df['totalwords'] = df['message'].str.split().str.len()
    words_counts = df.groupby('totalwords').count()['message']
    words_counts  = words_counts[:100]
    words_number = list(words_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'distribution of genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "genre"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=categories_names,
                    y=categories_list
                )
            ],

            'layout': {
                'title': '36 Categories counts',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=words_number,
                    y=words_counts
                )
            ],

            'layout': {
                'title': 'message words distribution',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "number of words in a message"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    #app.run(host='75.182.141.71', port=3001, debug=True)
    app.run(host='0.0.0.0', port=3001, debug=True)



if __name__ == '__main__':
    main()