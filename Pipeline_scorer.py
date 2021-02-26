#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report , accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import pandas as pd


# In[3]:


class Pipeline_scorer:
    '''
    This function trains a selected model with GridSearchCV and perform a pipeline for vectorizing and transform
    the text data. It outputs the best parameters, the f1-score, the precision and the recall for every outout
    of the MultiOutputClassifier.Also it gives the mean for the whole results and the f_1_score averaged by
    the function.
    
    '''
    
        def __init__ (self, parameters_grid,classifier):
            '''
            Initiazalizes the class. 
            Takes the parameters_grid for the Cross_Validation and the classifier that the MultiOutputClassifier
            will use.
            
            Parameters:
            parameters_grid : dict,
                parameters for the Cross_validation
            classifier: func,
                function for the MultiOutputClassifier
            
            '''
            
            self.parameters_grid = parameters_grid
            self.classifier = classifier
        
        def pipeline(self, X_train,y_train,X_test,y_test,jobs,
                     score=f1_score, scoring ='f1_micro',transformer = TfidfTransformer()):
            
                        '''
            Make a pipeline using the splited data (train, test) with Feature Union on Count_vectorizer and the transformer selected by the user
            for the text pipeline (with TfidfTransformer() as default). For the classifier it uses
            the MultiOutputClassifier with the transformer selected.                     
                                   
            Takes the parameters_grid for the Cross_Validation and the classifier that the MultiOutputClassifier
            will use.
            
            Parameters:
            X_train: array-like of shape (n_samples, n_features),
                The train data to fit. Can be for example a list, or an array.
            y_train: array-like of shape (n_samples,) or (n_samples, n_outputs), default=None),
                The train data to fit. Can be for example a list, or an array.
            X_test: array-like of shape (n_samples, n_features),
                The test data to fit. Can be for example a list, or an array.
            y_test: array-like of shape (n_samples,) or (n_samples, n_outputs), default=None),
                The test data to fit. Can be for example a list, or an array.
            jobs: int, default=None,
                Number of jobs to run in parallel.
                Training the estimator and computing the score are parallelized over the cross-validation splits.
                None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.
            score=func, default = f1_score(),
                The evaluation metrics used for the whole model.
            scoring = callable, list, tuple, or dict, default='f1_micro'str,
                Strategy to evaluate the performance of the cross-validated model on the test set.
                For more information see the sklearn.model_selection.cross_validate documentation
            transformer = func, default = TfidfTransformer(),
                The function used inside the text pipeline for vectorizing the text data.
            
            Returns a fitted cross valided and prints its best parameters. 
            '''
        
            pipeline = Pipeline([
                ('features', FeatureUnion([

                    ('text_pipeline', Pipeline([
                        ('vect', CountVectorizer()),
                        ('transformer', transformer)
                    ]))            
                ])),
                ('clf', MultiOutputClassifier(self.classifier))
            ])

            cv = GridSearchCV(pipeline, param_grid=self.parameters_grid, scoring=scoring, n_jobs=jobs)
            cv.fit(X_train, y_train)
            self.y_pred = cv.predict(X_test)
            self.y_test = y_test
            print(cv.best_params_)

    
        def report(self, average = 'weighted'):
            
            '''
            Uses the fitted model in the pipeline function to make a report with the scores of each output
            of the MultiOutputClassifier and for the whole dataset.
            
            Parameters:
            average = str or None, default=’weighted',
                The average parameter for the score function selected.
            
            Returns a printed dataframe for each variable, a dataframe for the raw mean of the variables
            and a print statement for the score and average selected by the user.
            '''
            
            
            
            report = {}
            for n, col in enumerate(self.y_test.columns):
                output = classification_report(self.y_test[col], self.y_pred[:,n], output_dict=True)
                report[col] = {}
                for i in output:   
                    if i == 'accuracy':
                        break
                    report[col]['f1_' + i] = output[i]['f1-score']
                    report[col]['precision_' + i] = output[i]['precision']
                    report[col]['recall_' + i] = output[i]['recall']

            report_df = pd.DataFrame(report).transpose()
            report_df = report_df[report_df.columns.sort_values()]
            report_df_mean = report_df.mean()

            print("Table for each column:")
            print (report_df)
            print('\n')
            print('mean of results:')
            print(report_df_mean)
            print('\n')
            print('f1-score ({}): {}'.format(average, score(self.y_test, self.y_pred, average=average)))


# In[ ]:




