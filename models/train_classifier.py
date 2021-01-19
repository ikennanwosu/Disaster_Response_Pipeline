# import libraries
import os
import sys
import re
import pandas as pd
from sqlalchemy import create_engine
import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    # Create a new Engine instance
    engine = create_engine('sqlite:///'+ database_filepath)
    
    # Read data from DisasterMessages SQL table
    df = pd.read_sql_table('DisasterMessages', engine) 
    
    # Select the column names for the categories of the disaster messages
    category_names = df.drop(['id', 'message', 'original','genre'], axis=1).columns.tolist()
    
    # Select the training and test data
    X = df.message
    Y = df[category_names]
    
    return X, Y, category_names


def tokenize(text):
    # Remove punctuation characters and convert text to lowercase
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Return a tokenized copy of the text
    tokens = word_tokenize(text)
    
    # Initialize the WordNet Lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # lemmatize and remove stop words
    lemmatized_tokens = [lemmatizer.lemmatize(w) for w in tokens 
                         if w not in stopwords.words("english")]
    
    return lemmatized_tokens


def build_model():
    # Create data pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=10)))
    ])
    
    parameters = {
        'tfidf__sublinear_tf': [True, False]#,
        #'vect__ngram_range':((1, 1), (1, 2)),
        #'clf__estimator__criterion' : ['gini'],
        #'clf__estimator__n_estimators': [10, 20],
    }

    # create grid search object
    grid_search = GridSearchCV(pipeline, param_grid=parameters)
    
    return grid_search


def train(X, Y, model, category_names):
    # train test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    # fit model
    print('Training model...')
    model.fit(X_train, Y_train)

    # output model test results
    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    return model


def evaluate_model(model, X_test, y_test, category_names):  
    
    # predict on test data
    y_pred = model.predict(X_test)
    
    for i in range(len(category_names)):
        accuracy = accuracy_score(y_test.iloc[:, i].values, y_pred[:, i])
        print('Category: {} '.format(category_names[i]))
        print('Accuracy: {}%\n'.format(int(round(100*accuracy,0))))
        print(classification_report(y_test.iloc[:, i].values, y_pred[:, i]))
        print("====================================================")


def save_model(model, model_filepath):
    """
        Exports the final model as a pickle file
        
        args:
            model: the final trained model
            model_filepath: directory to save the model
    """
    pickle.dump(model, open(os.path.join(model_filepath), 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        model = build_model()
        model = train(X, Y, model, category_names)        

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')
        

if __name__ == '__main__':
    main()