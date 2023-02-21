import sys
import pickle
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download("stopwords", quiet=True)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    '''Loads Data From DataBase

    Args:
        database_filepath (str): Path of the sqlite Db used for Storage

    Returns:
        X: Messages
        Y: Category Values
        categories: Category Names
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('Messages', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    Y.replace({'related':{2:1}}, inplace=True)
    Y.drop('child_alone', axis=1, inplace=True)
    categories = Y.columns
    return X, Y, categories


def tokenize(text):
    '''Tokenizes Text Data

    Args:
        text (str): Messages Raw Text Data

    Returns:
        words (list): list of tokens
    '''
    stemmer = PorterStemmer()
    #remove URLs taken from 
    # https://gist.github.com/MrEliptik/b3f16179aa2f530781ef8ca9a16499af
    text = re.sub(r"http\S+", "", text)
    # Convert to lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Split string into words
    words = text.split()
    # Remove stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    # stem
    words = [stemmer.stem(w) for w in words]
    return words


def build_model(model=LogisticRegression(random_state=123), param_grid=None):
    '''Build Model Pipeline

    Args:
        model (sklearn.Estimator): sklearn pipeline compatible Estimator
        param_grid (dict): params to pass to the pipeline

    Returns:
         (sklearn.GridSearchCV): if param grid_dict arg has been specified
         (sklearn.Pipeline): if param grid_dict arg hasn't been specified
    '''
    pipeline_clf = make_pipeline(CountVectorizer(tokenizer=tokenize),
                                 TfidfTransformer(),
                                 MultiOutputClassifier(model)
                                 )
    if param_grid:
        cv = GridSearchCV(pipeline_clf, param_grid=param_grid,
                          cv=5, verbose=4, n_jobs=-1
                          )
        return cv
    else:
        return pipeline_clf


def evaluate_model(model, X_test, Y_test, category_names):
    '''Prints Classification Report

    Args:
        model (Pipeline or GridSearchCV): Takes A fitted Pipeline
        or GridSearchCV object
        X_test (array): Text Messages not seen by model
        Y_test (array or DataFrame): Categories associated with Messages
        category_names (list): category column names

    Returns:
        words (list): list of tokens
    '''
    y_pred = model.predict(X_test)
    report = classification_report(Y_test.values, y_pred,
                                   target_names=category_names
                                   )
    print(report)


def save_model(model, model_filepath):
    '''Save Trained Model

    Args:
        model (sklearn.Estimator): Fitted sklearn pipeline compatible Estimator
        model_filepath (str): Path to save model

    Returns:
         None: Pickles the trained model ti given path
    '''
    with open(model_filepath, 'wb') as clf:
        pickle.dump(model, clf)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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