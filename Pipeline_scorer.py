#!/usr/bin/env python
# coding: utf-8

# In[26]:


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report , accuracy_score
from sklearn.model_selection import GridSearchCV
import pandas as pd

# In[28]:


class Pipeline_scorer:
    
        def __init__ (self, parameters_grid,classifier):
            self.parameters_grid = parameters_grid
            self.classifier = classifier
        
        def pipeline(self, X_train,y_train,X_test,y_test,jobs,
                     score=f1_score,average = 'weighted', scoring ='f1_micro',transformer = TfidfTransformer()):
        
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

    
        def report(self, average = 'weighted', score=f1_score):
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
            print('{} ({}): {}'.format(score, average, score(self.y_test, self.y_pred, average=average)))


# In[ ]:




