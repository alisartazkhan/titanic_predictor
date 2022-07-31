'''
Name: Ali Sartaz Khan
Description: Project uses TensorFlow to analyse Titanic data and take in a set of
inputs that are used describe various characteristics about a person, and then using
the Titanic data it will predict the likelihood of that individual making it to the
rescue boat using Linear Regression.
'''
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import tensorflow as tf
import tensorflow.compat.v1.feature_column as fc
import random

def evaluation_set():
    '''
    Function used to take in input regarding the characteristics of the person that
    you want to make a prediction on regarding the probability of that individual
    making it to the rescue boat.
    '''
    header = ["survived", 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare',
              'class', 'deck', 'embark_town', 'alone']
    soln = [str(random.randint(0,1))]
    soln.append(input('Sex:\n'))
    soln.append(input('Age:\n'))
    soln.append(input('Number of Siblings/Spouses:\n'))
    soln.append(input('Parch (0-5):\n'))
    soln.append(input('Fare:\n'))
    soln.append(input('Class (First/Second/Third):\n'))
    soln.append(input('Deck Number (A-F or Unknown):\n'))
    soln.append(input('Town (Southampton/Cherbourg/Queenstown):\n'))
    soln.append(input('Is the passenger travelling alone? (y/n):\n'))

    data = pd.DataFrame([soln], columns = header)
    data.to_csv("TestData.csv")
    result = pd.read_csv('TestData.csv')
    return result


def featureColumns(TRAINING_SET):
    '''
    Function creates feature columns using tensor flow.
    TRAINING_SET: Pandas object that reads data from csv file
    returns: feature_columns [a list containing all the different features from training data set]

    '''
    CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
    NUMERIC_COLUMNS = ['age', 'fare']
    feature_columns = []
    for feature_name in CATEGORICAL_COLUMNS:
        vocabulary = TRAINING_SET[feature_name].unique()  # gets a list of all unique values from given feature column
        feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

    for feature_name in NUMERIC_COLUMNS:
        feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
    return feature_columns


def make_input_fn(data_df, label_df, num_epochs=15, shuffle=True, batch_size=32):
    '''
    This function creates a tf.data.Dataset object with data and it's label and returns it.
    '''
    def input_function():
         # create tf.data.Dataset object with data and its label
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)  # randomize order of data
        # split dataset into batches of 32 and repeat process for number of epochs
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds  # return a batch of the dataset
    return input_function  # return a function object for use

def output(result, passengerINFO):
    '''
    Function uses the given parameters and prints out the output of the study.
    result: integer representing the probabilty of getting on the rescue boat
    passengerINFO: first row of TestData.csv containing information regarding the
    passenger
    return: N/A
    '''
    print("PASSENGER INFORMATION:")
    count = 0
    dictionary = dict(passengerINFO)
    print("-"*36)
    for category, res in dictionary.items():
        if count == 0 or count == len(dictionary):
          count +=1
          continue
        x = '| {:>17} :'
        y = "{:>12} |"
        if category == "n_siblings_spouses":
            category = "Siblings/Spouses"
        print(x.format(category), y.format(res))
        count +=1
    print("-"*36)
    print()
    print("This individual has", str(round(result*100, 1)) + "%", "chance of getting on the rescue boat!")


def main():
  # Load dataset.
  TRAINING_SET = pd.read_csv('TrainingData.csv') # training data
  y_train = TRAINING_SET.pop('survived')

  evaluationSet = evaluation_set()
  y_eval = evaluationSet.pop('survived')
  clear_output()
  feature_columns = featureColumns(TRAINING_SET)
  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
  training_fn = make_input_fn(TRAINING_SET, y_train)
  evaluation_fn = make_input_fn(evaluationSet, y_eval, num_epochs=1, shuffle=False)
  # create a linear estimator by passing the feature columns we created earlier
  linearEstimate = tf.estimator.LinearClassifier(feature_columns)
  linearEstimate.train(training_fn)  # train the estimator
  clear_output()  # clears console output
  # the result variable is simply a dict of stats about our model
  result = list(linearEstimate.predict(evaluation_fn))
  output(result[0]["probabilities"][1], evaluationSet.loc[0])


main()