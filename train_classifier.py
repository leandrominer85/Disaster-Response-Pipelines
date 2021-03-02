import sys
import numpy as np
import pandas as pd
import sqlite3
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline , FeatureUnion
from sklearn.metrics import classification_report, f1_score, make_scorer, fbeta_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pickle

def load_data(database_filepath = '/home/worspace/data/DisasterResponse.db'):
 
    # load data from database
    conn = sqlite3.connect(f"{database_filepath}")

    df = pd.read_sql('SELECT * FROM disaster', con=conn)
    

    #Remove child alone as it has all zeros only
    df = df.drop(['child_alone'],axis=1)
    #Dropping the related rows with value  = 2
    df = df[df['related'] != 2]
    X = df['message']
    Y = df.loc[:,'related':'direct_report']
    categories = Y.columns.tolist()
    
    return X,Y,categories


def tokenize(text):
    tokens = word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()).replace("  ",""))
    words = [w for w in tokens if w not in stopwords.words("english")]
    

    # Reduce words to their stems
    stemmed = [PorterStemmer().stem(w) for w in words]
    
    # Reduce words to their stems
    
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in stemmed]
    
    return lemmed 


def build_model():

    pipeline = Pipeline([
            ('features', FeatureUnion([

                ('text_pipeline', Pipeline([
                    ('vect', CountVectorizer()),
                    ('transformer', TfidfTransformer())
                ]))
            ])),
            ('clf', MultiOutputClassifier(AdaBoostClassifier(algorithm = 'SAMME.R', learning_rate = 1, n_estimators = 50)))
        ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))
    print("Fbeta score:", fbeta_score(Y_test, y_pred, beta=2, average="weighted"))



def save_model(model, model_filepath):
        pickle.dump(model, open(model_filepath, "wb"))


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
