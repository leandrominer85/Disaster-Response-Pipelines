# Disaster Response Pipeline Project
Project for the Udacity Nanodegree course.
The goal of this project is to apply data engineering skills to analyze a dataset provided by Figure Eight containing over 30000 real messages sent to disaster response organizations to ultimately build a model that classifies disaster messages for an API. The web application can therefore be effective for real-world disaster scenarios. 

 The project is divided into three sections: 

 - **Data Processing**: build an ETL (Extract, Transform, and Load) pipeline that processes messages and category data from CSV file, and load them into an SQLite database - which our ML pipeline will then read from to create and save a multi-output supervised learning model.

 - **Machine Learning Pipeline**: split the data into a training set and a test set. Then, create an ML pipeline that uses NLTK, as well as GridSearchCV to output a final model that predicts message classifications for the 36 categories (multi-output classification)
  
 - **Web development**: deploy a web application that classifies messages.
 

## Software & Libraries

The project uses Python 3 and the following libraries:

-   [NumPy](http://www.numpy.org/)
-   [Pandas](http://pandas.pydata.org/)
-   [nltk](https://www.nltk.org/)
-   [scikit-learn](http://scikit-learn.org/stable/)
-   [sqlalchemy](https://www.sqlalchemy.org/)
-   [sqlite3](https://www.sqlite.org/index.html)
-   [flask](https://flask.palletsprojects.com/en/1.1.x/)



## Data
The dataset is provided by Figure Eight and consists of: 
-   **disaster_categories.csv**: message categories
-   **disaster_messages.csv**: multilingual disaster response messages


## Instructions:

1.  Run the following commands in the project's root directory to set up your database and model.
    
    -   To run ETL pipeline that cleans data and stores in database  `python process_data.py messages.csv categories.csv DisasterResponse.db`
    -   To run ML pipeline that trains classifier and saves  `python models/train_classifier.py DisasterResponse.db models/classifier.pkl`
2.  Run the following command in the app's directory to run your web app.  `python app/run.py`
    
3.  Open another terminal, run  `env|grep WORK`. You'll see the following output WORKSPACEDOMAIN=udacity-student-workspaces.com WORKSPACEID=view6914b2f4 Now, use the above information to open  [https://view6914b2f4-3001.udacity-student-workspaces.com/](https://view6914b2f4-3001.udacity-student-workspaces.com/)  (general format -  [https://WORKSPACEID-3001.WORKSPACEDOMAIN/](https://workspaceid-3001.workspacedomain/))

## Results 

The web application:


![Screen Shot 2021-01-11 at 01.03.14.png](https://www.dropbox.com/s/29fidajr646o2pn/01.png?dl=0)


When a disaster message is submitted and the Classify Message button is clicked, the app shows how the message is classified by highlighting the categories in green. 
For the suggested message "we are more than 50 people on the street. Please help us find tent and food" we get the following categories: 


![Screen Shot 2021-03-01 at 16.15.41.png](https://www.dropbox.com/s/xqraqb32rj5r9xv/02.png?dl=0)



## Licensing and Acknowledgements
Thanks Udacity for the course and [Figure Eight](https://appen.com/resources/datasets/) for the dataset.
