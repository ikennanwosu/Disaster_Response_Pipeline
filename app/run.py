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
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterMessages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # To plot the count of messages per genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # To plot the percentage of the mean message categories per category
    category_names = df.columns[4:].tolist()
    df_new = df.melt(id_vars='message', value_vars=category_names)
    df_new.columns = ['message', 'category', 'response']
    category_mean = df_new.groupby('category')['response'].mean()
    category_precent_mean = round(100*category_mean,0).values.astype(int)
    
    # To plot the percentage of the mean messages categories per genre
    df_genre = df.melt(id_vars='genre', value_vars=category_names)
    df_genre.columns = ['genre', 'category', 'values']
    df_genre_mean = df_genre.groupby(['genre'])['values'].mean().reset_index(name='mean_value')
    
    
    graphs = [
        {
          'data': [
                Bar(
                    x=category_names,
                    y=category_precent_mean
                )
            ],

            'layout': {
                'title': 'Percentage Distribution of Disaster Message Categories per Category',
                'yaxis': {
                    'title': "Percentage Mean (%)"
                },
                'xaxis': {
                    'title': "Message Category"
                }
            }
        },
       {
          'data': [
                Bar(
                    x=df_genre_mean['genre'].values,
                    y=df_genre_mean['mean_value'].values
                )
            ],

            'layout': {
                'title': 'Proportion of Disaster Message Categories per Genre',
                'yaxis': {
                    'title': "Proportion"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()