import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
       #Read the dfs
    df1 = pd.read_csv(messages_filepath)
    df2 = pd.read_csv(categories_filepath)
    
    #Merge the dfs
    df = df1.merge(df2, on='id')
    
    return df

def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    #Creating a list of the columns names with the first row value and then extract the final and assign to the DF 
    col_list = list(categories.iloc[0].values)
    categories.columns = [x.split("-")[0] for x in col_list] 
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] =  categories[column].str.split('-',expand=True)[1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    # drop the original categories column from `df`
    df.drop("categories", inplace = True, axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(['id'],inplace=True)
    
    return df


def save_data(df, database_filename = 'disaster'):
    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('{}'.format('disaster'), engine, index=False)
   
   


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
