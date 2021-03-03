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
from nltk.stem.porter import PorterStemmer

def load_data(database_filepath = '/home/worspace/data/DisasterResponse.db'):

    '''
    INPUTS:
    - Filepath to the database
    
    OUTPUT:
    - category_names: categories names from the dataframe, X: text messages from the dataframe, Y: column names (categories)

    '''
 
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
    from nltk.corpus import stopwords # This import is here due to a error:
     # (https://stackoverflow.com/questions/44911539/pickle-picklingerror-args0-from-newobj-args-has-the-wrong-class-with-hado)


    '''
    INPUTS:
    - Text (str) input
    
    OUTPUT:
    - Tokenized and lemmed text 

    '''

    tokens = word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()).replace("  ",""))
    words = [w for w in tokens if w not in stopwords.words("english")]
    
    
    # Reduce words to their stems
    
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    return lemmed 


def build_model():


    '''
    
    OUTPUT:
    - A pipeline that uses cross validation with AdaboostClassifier after the text transformation with
     CountVectorizer(tokenizer=tokenize) and TfidfTransformer(). The scoring metric is the f1_micro

    '''

    pipeline = Pipeline([
            ('features', FeatureUnion([

                ('text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('transformer', TfidfTransformer())
                ]))
            ])),
            ('clf', MultiOutputClassifier(AdaBoostClassifier()))
        ])


    learning_rate =[0.5,1 ]
    n_estimators = [50,100] 
    algorithm = ['SAMME', 'SAMME.R']
    parameters_grid    = dict(clf__estimator__learning_rate= learning_rate,
                              clf__estimator__n_estimators=n_estimators,clf__estimator__algorithm=algorithm)
    pipe_cv = GridSearchCV(pipeline, param_grid=parameters_grid, scoring='f1_micro', n_jobs=-1)





    return pipe_cv


def evaluate_model(model, X_test, Y_test, category_names):


    '''
    INPUTS:
    - The saved model
    - The array for the train test
    - The dataframe for the train test
    
    OUTPUT:
    - Prints the classification report for model  and Fbetascore

    '''    
    
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))
    print("Fbeta score:", fbeta_score(Y_test, y_pred, beta=2, average="weighted"))



def save_model(model, model_filepath):

    '''
    INPUTS:
    - The pipeline model
    - The filepath to save the model
    
    
    OUTPUT:
    - saved model

    ''' 
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
