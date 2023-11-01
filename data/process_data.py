import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    INPUT:
    messages_filepath - path to messages csv file
    categories_filepath - path to categories csv file
    
    OUTPUT:
    df - Merged data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on = 'id')
    return df


def clean_data(df):
    """
    Clean the merged data.
    
    Parameters:
        df (DataFrame): Merged data.
        
    Returns:
        DataFrame: Cleaned data.
    """
    # Create dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    
    # Use the first row of the categories dataframe to extract new column names for categories
    category_colnames = categories.iloc[0].apply(lambda x: x[:-2])
    
    # Rename the columns of 'categories'
    categories.columns = category_colnames
    
    # Convert category values to numbers 0 or 1
    categories = categories.apply(lambda x: x.str[-1]).astype(int)
    
    # Drop the original categories column from 'df'
    df = df.drop('categories', axis=1)
    
    # Concatenate the original dataframe with the new 'categories' dataframe
    df = pd.concat([df, categories], axis=1)
    
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    # Drop 'child_alone' column as it has only 0 values
    df = df.drop('child_alone', axis=1)
    
    # Replace '2' with '1' in the 'related' column
    df['related'] = df['related'].replace(2, 1)
        
    return df


def save_data(df, database_filename):
    """
    Save the cleaned data to a SQLite database.
    
    Parameters:
        df (DataFrame): Cleaned data.
        database_filename (str): Filename for the SQLite database (*.db file).
        
    Returns:
        None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Disaster_Response', engine, index=False, if_exists='replace')


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