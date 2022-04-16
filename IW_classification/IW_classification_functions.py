# -*- coding: utf-8 -*-
"""
last update 15/03/22
@author: Tonatiuh Hernández-del-Toro (tonahdztoro@gmail.com)

Code with the functions to run the code IW_classification.py in the paper 
"AU-NN: ANFIS Unit Neural Network"

arXiv: 
"""

# To avoid innecesary log messages
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

# Libraries needed
import tensorflow as tf
import random
import numpy as np
import mat4py

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from collections import Counter




def merge_dataset(Dataset, cut):
    """
    Function to create a fold with a train and test set for all subjects except [2,3,5] with a given dataset

    Parameters
    ----------
    Dataset : The dataset containing all the words
    cut : the cut (1-30) to take the % of words for train set and % of words for test set

    Returns
    -------
    X_train : The set of instances as a matrix (no_instances, no_features)
    y_train : The classes of the train set as a vector
    X_y_test : The test set including instances and classes
    """
    
    # Initialize the sets
    X_train = np.zeros((1,70))
    y_train = np.zeros((1,1))
    X_y_test = {'Instances': [], 'Classes': []}
    
    # Run for every word
    for word in range(len(Dataset)):
        
        # Select the word
        Word = Dataset[word]
        Instances = np.array(Word['Instance'],dtype=object)
        Classes = np.array(Word['Class'],dtype=object)
        
        # Randomize the instances in the word
        c = list(zip(Instances, Classes))
        random.shuffle(c)
        Randomized_Instances, Randomized_Classes = zip(*c)
        
        # Select the words that belong to the train and test set
        Words_instances_train = Randomized_Instances[:cut]
        Words_instances_test = Randomized_Instances[cut:]
        Words_classes_train = Randomized_Classes[:cut]
        Words_classes_test = Randomized_Classes[cut:]
        
        # Append the words that belong to the train set
        for instance in range(len(Words_instances_train)):
            Instance_train = np.array(Words_instances_train[instance])
            Class_train = np.array(Words_classes_train[instance])
            
            X_train = np.concatenate((X_train,Instance_train),axis=0)
            y_train = np.concatenate((y_train,Class_train),axis=0)
            
        # Append the words that belong to the test set
        for instance in range(len(Words_instances_test)):
            Instance_test = np.array(Words_instances_test[instance])
            Class_test = np.array(Words_classes_test[instance])
            
            X_y_test['Instances'].append(Instance_test)
            X_y_test['Classes'].append(Class_test)
            
          
    X_train = X_train[1:,:]
    y_train = y_train[1:,:]
    
    
    return X_train, y_train, X_y_test


def merge_dataset_short_words(Dataset, cut):
    """
    Function to create a fold with a train and test set for all subjects [2,3,5] with a given dataset

    Parameters
    ----------
    Dataset : The dataset containing all the words
    cut : The cut (1-30) to take the % of words for train set and % of words for test set

    Returns
    -------
    X_train : The set of instances as a matrix (no_instances, no_features)
    y_train : The classes of the train set as a vector
    X_y_test : The test set including instances and classes
    """
    
    # Initialize the sets
    X_train = np.zeros((1,70))
    y_train = np.zeros((1,1))
    X_y_test = {'Instances': [], 'Classes': []}
    
    # Run for every word
    for word in range(len(Dataset)):
        
        # Select the word
        Word = Dataset[word]
        Instances = np.array(Word['Instance'],dtype=object)
        Classes = np.array(Word['Class'],dtype=object)
        
        # Randomize the instances in the word
        c = list(zip(Instances, Classes))
        random.shuffle(c)
        Randomized_Instances, Randomized_Classes = zip(*c)
        
        # Select the words that belong to the train and test set
        Words_instances_train = Randomized_Instances[:cut]
        Words_instances_test = Randomized_Instances[cut:]
        Words_classes_train = Randomized_Classes[:cut]
        Words_classes_test = Randomized_Classes[cut:]
        

        # Append the words that belong to the train set
        for instance in range(len(Words_instances_train)):
          if Words_classes_train[instance] != []:
            if len(Words_instances_train[instance]) == 70:
              Instance_train = np.reshape(Words_instances_train[instance], (1,70))
              Class_train = np.reshape(Words_classes_train[instance],(1,1))
              
              X_train = np.concatenate((X_train,Instance_train),axis=0)
              y_train = np.concatenate((y_train,Class_train),axis=0)

            else:
              Instance_train = np.array(Words_instances_train[instance])
              Class_train = np.array(Words_classes_train[instance])
              
              X_train = np.concatenate((X_train,Instance_train),axis=0)
              y_train = np.concatenate((y_train,Class_train),axis=0)


        # Append the words that belong to the test set
        for instance in range(len(Words_instances_test)):
          if Words_classes_test[instance] != []:
            if len(Words_instances_test[instance]) == 70:
              Instance_test = np.reshape(Words_instances_test[instance], (1,70))
              Class_test = np.reshape(Words_classes_test[instance],(1,1))
              
              X_y_test['Instances'].append(Instance_test)
              X_y_test['Classes'].append(Class_test)

            else:
              Instance_test = np.array(Words_instances_test[instance])
              Class_test = np.array(Words_classes_test[instance])
              
              X_y_test['Instances'].append(Instance_test)
              X_y_test['Classes'].append(Class_test)


    X_train = X_train[1:,:]
    y_train = y_train[1:,:]
    
    
    return X_train, y_train, X_y_test


def create_fold(subject,cut):
    """
    Function to create a fold with a train and test set given any subject

    Parameters
    ----------
    subject : The subject we want to create a fold
    cut : The cut (1-30) to take the % of words for train set and % of words for test set


    Returns
    -------
    X_train : The set of instances as a matrix (no_instances, no_features)
    y_train_hot : The classes of the train set in a one_hot fashion
    X_y_test : The test set including instances and classes
    """
    
    # Load the dataset
    file_path = 'Datasets/S'  + str(subject + 1) + '.mat'
    mat = mat4py.loadmat(file_path);
    Dataset = mat['Dataset']['Word'];
    
    # Select the method according to the subject number
    if subject in [2,3,5]:
      X_train, y_train, X_y_test = merge_dataset_short_words(Dataset, cut)
    else:
      X_train, y_train, X_y_test = merge_dataset(Dataset, cut)
      
    # Create the y_train_hot : The classes of the train set in a one_hot fashion
    depth = 5
    y_train_hot = tf.one_hot(y_train, depth, dtype=tf.float64)
    y_train_hot = tf.transpose(tf.reshape(y_train_hot[:,:,:],[y_train_hot.shape[0],depth]))
    
    return X_train, y_train_hot, X_y_test





def evaluate_subject_fold(X_train, y_train_hot, X_y_test, model):
    """
    Function that trains a given model with a given train set and test it with a given test set

    Parameters
    ----------
    X_train : The set of instances as a matrix (no_instances, no_features)
    y_train_hot : The classes of the train set in a one_hot fashion
    X_y_test : The test set including instances and classes
    model : The AU_NN model

    Returns
    -------
    The accuracy of the model on the given test set X_y_test

    """
    
    Learning_rates = [0.03,0.01,0.003]
    epochs = 500
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    
    for learning_rate in Learning_rates:
    
        model.fit(X_train,y_train_hot,epochs,learning_rate)

        test_pred = np.array([])
        y_test_numpy = np.array([])

        X_test = X_y_test['Instances']
        y_test = X_y_test['Classes']

        for instance in range(len(X_test)):
            X_major = X_test[instance]
            X_major = scaler.transform(X_major)
            
            pred_major_numpy = model.predict(X_major)
            pred_major_numpy = np.argmax(pred_major_numpy,axis=0)
            pred_major_numpy = np.clip(pred_major_numpy,0,4)

            pred = Counter(pred_major_numpy).most_common(1)[0][0]
            test_pred = np.append(test_pred,pred)
            
            y_major = y_test[instance].ravel()
            y = Counter(y_major).most_common(1)[0][0]
            y_test_numpy = np.append(y_test_numpy, y)
    
    return accuracy_score(y_test_numpy, test_pred)


