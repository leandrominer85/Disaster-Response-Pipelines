def load_data (data1,data2,db_name='Disaster',table_name='Disaster'):

    '''
    This function receives two paths to data files in the .csv format (data1, data 2).
    After that it merges both dataframes in the common column ('id'). Uses the second dataframe
    (with categories column) to create a new dataframe with columns names based on the first row name
    (the string before the "-" character). The values of the columns of this new dataframe are the numeric part
    of the end of each row (the number after the "-" character). Then it concats this dataframe with
    the one merged and drop the old "categories" column. Finally it drops the "id" duplicates and saves the
    dataframe on a table in a database.
    The user can change the database and table names in the function db_name and table_name.
    
    '''
    
    
    #Read the dfs
    df1 = pd.read_csv(data1)
    df2 = pd.read_csv(data2)
    
    #Merge the dfs
    df = df1.merge(df2, on='id')
    
    
    # create a dataframe of the 36 individual category columns
    categories = df2['categories'].str.split(';', expand=True)

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
    
    engine = create_engine('sqlite:///{}.db'.format(db_name))
    df.to_sql('{}'.format(table_name), engine, index=False)
    
    return