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
    """
    Loads the processed dataset from an SQL database
    
    args:
        database_filepath: file path to the database
        
    return:
        X: the messages received during the disaster events
        Y: the categories/labels of the messages
        category_names: the names of each category to be predicted       
    """
    
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
    """
    Splits the entire messages received during the 
    disaster event into smaller units, such as individual 
    words or terms.
    It removes punctuation characters and convert the text
    to lowercases, finally removing all non-meaningful stopwords.
    
    args:
        text: (string) the received disaster messages
        
    return:
        lemmatized_tokens: tokenized list of messages
    
    """
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
    """
    Builds a model by wrapping the following in a pipeline object;
    CountVectorizer: which transforms a given text into a vector 
                     on the basis of the frequency (count) of each 
                     word that occurs in the entire text.
    TfidfTransformer: transforms a count matrix from the CountVectorizer 
                      to a normalized tf or tf-idf representation.
                      tf: term frequency
                      idf: inverse document frequency.
                      The goal of using tf-idf instead of the raw frequencies 
                      of occurrence of a token in a given document is to scale 
                      down the impact of tokens that occur very frequently in a 
                      given corpus and that are hence empirically less informative 
                      than features that occur in a small fraction of the training corpus.
                      It takes the tokenizer function as input.
    MultiOutputClassifier: used to extend classifiers that do not natively support the 
                           multi-target classification problem. It is achieved by fitting
                           one classifier per target/class.
    RandomForestClassifier: creates decision trees on randomly selected data samples, 
                            gets prediction from each tree and selects the best solution 
                            by means of voting.
    GridSearchCV: performs an exhaustive search over specified hyperparameter values (eg.
                  a parameter dictionary) for an esimator, by fitting and performing prediction
                  using cross-validation method. The hyperparameter values with best performance
                  are chosen for the final model.
      
    return:
        grid_search: an object to be used to search for optimal values for the given model
  
    
    """
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
    """
    Fits the specified model/estimator to the training
    data, and performs and outputs evaluation results of the 
    trained model on the test data.
    The evaluation results are as follows:
        Accuracy: proportion of true results among the total 
                  number of cases examined. 
        Precision: proportion of predicted positives are truely
                   positive
        Recall:  proportion of actual positives that are correctly
                 classified
        F1-score: measure of a model's acuracy as a weighted average
                  between the precision and recall.
    
    args:
        X: the predictor/features/messages 
        Y: the target/labels of the categories
        category_names: list of message categories
        
    return:
        model: trained model
    
    """
    # train test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    # fit model
    print('Training model...')
    model.fit(X_train, Y_train)

    # output model test results
    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    return model


def evaluate_model(model, X_test, Y_test, category_names):  
    """
    Returns evaluation results of the trained model on 
    the test datasets.
    
    args:
        model: the specified trained model
        X_test: test set of the messages
        Y_test: test set of the target/labels of the messages
        category_names: the names of each category to be predicted       
    """
    
    # predict on test data
    Y_pred = model.predict(X_test)
    
    for i in range(len(category_names)):
        accuracy = accuracy_score(Y_test.iloc[:, i].values, Y_pred[:, i])
        print('Category: {} '.format(category_names[i]))
        print('Accuracy: {}%\n'.format(int(round(100*accuracy,0))))
        print(classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
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