# import libraries
import sys
import numpy as np
import pandas as pd
import re
import pickle
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import FunctionTransformer



def load_data(database_filepath):
    """
    Load data from a SQLite database.

    Args:
        database_filepath (str): Filepath of the SQLite database.

    Returns:
        tuple: A tuple containing the following elements:
            - X (pandas.Series): The 'message' column of the loaded dataset.
            - Y (pandas.DataFrame): The target variables (categories) of the loaded dataset.
            - category_names (list): A list of category names.

    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("Disaster_Response", engine)
    X = df["message"]
    Y = df.drop(["id", "message", "original", "genre"], axis=1)
    category_names = Y.columns.tolist()
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize the text data.
    
    Args:
        text (str): The input text.
    
    Returns:
        list: A list of tokens.
    """
    # Check if the input is a string
    if isinstance(text, str):
        # Convert the text to lowercase
        text = text.lower()

        # Remove punctuation characters from the text and replace them with an empty space
        text = re.sub(r"[^a-zA-Z0-9]", " ", text)

        # Tokenize the text by splitting on whitespace
        tokens = text.split()

        # Remove stopwords
        tokens = [word for word in tokens if word not in stopwords.words("english")]

        # Lemmatize the tokens
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        return tokens
    else:
        # Return an empty list if the input is not a string
        return []


def build_model(rf=RandomForestClassifier()):
    """
    Build a machine learning model using a pipeline and perform hyperparameter tuning.

    Args:
        rf (estimator, optional): The random forest classifier estimator. Defaults to RandomForestClassifier().

    Returns:
        GridSearchCV: A GridSearchCV object that contains the pipeline with the best-performing hyperparameters.

    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(rf))
    ])
    parameters = {
        "clf__estimator__n_estimators": [50, 100],
        "clf__estimator__min_samples_split": [2, 3]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv
    


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Prints the classification report for each category based on the predictions made by the pipeline.

    Args:
        pipeline: The trained pipeline used for prediction.
        X_test: The test data used for prediction.
        Y_test: The true labels for the test data.
        category_names: A list of category names.

    Returns:
        None

    """
    # Predict the labels for the test data using the pipeline
    predict_y = model.predict(X_test)

    # Iterate over each category
    for i in range(len(category_names)):
        category = category_names[i]
        print(category)
        
        # Print the classification report for the current category
        print(classification_report(Y_test[category], predict_y[:, i]))


def save_model(model, model_filepath):
    """
    INPUT:
    model - ML model
    model_filepath - location to save the model
    
    OUTPUT:
    none
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


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