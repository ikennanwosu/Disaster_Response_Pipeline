import sys
import pandas as pd
from pandas.io import sql
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Reads in the datasets disaster_messages.csv and 
    disaster_categories.csv for the ETL process. 
    The disaster_messages.csv file contains the messages 
    sent during disaster events, while the disaster_categories.csv
    file contains the categories to which each message belongs.    
    
    args:
        messages_filepath: path to the disaster_messages.csv file
        categories_filepath: path  to the disaster_categories.csv file
    
    return:
        a merged dataset of both csv files
    
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    return pd.merge(messages, categories, on='id')


def clean_data(df):
    """
    Performs data cleaning and transformation of the 
    loaded and merged disaster messaages.
    
    The steps undertaken were splitting the category entires 
    into text and integers; where the text indicate the 
    category of the messages received and the integer/labels (0, 1) 
    indicate whether or not each message relates to a category or not.
    
    args:
        df: (dataframe) the messages sent during the disasters events
    
    return:
        cleaned and transformed dataset
    
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.loc[0]
    
    # extract a list of new column names for categories
    category_columns = row.apply(lambda x: x.split('-')[0]).values.tolist()
    
    # assign the list to catergories to the category columns
    categories.columns = category_columns
    
    # extract the last numberic character from the each column
    # entry and convert to integer
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # concatenate the original dataframe, `df`, with the new 
    # `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(keep='first', inplace=True)
    
    return df


def save_data(df, database_filename):
    """
    Saves the clean and transformed data to
    an SQL database.
    
    args:
        df: (dataframe) the cleaned and transformed dataset    
    """
    
    # create an SQLAlchemy engine object
    engine = create_engine('sqlite:///' + database_filename)
    
    #     sql.execute('DROP TABLE IF EXISTS %s'%DisasterMessages, engine)
    #     sql.execute('VACUUM', engine)
    
    # write the records stored in the dataframe `df` to an SQL table
    df.to_sql('DisasterMessages', engine, index=False)


def main():
    """
    Performs the entire ETL process - Load, Extract and Saves the data

    """
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