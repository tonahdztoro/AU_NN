# -*- coding: utf-8 -*-
"""
last update 15/03/22
@author: Tonatiuh Hernández-Del-Toro (tonahdztoro@gmail.com)

Code with the functions to run the code Incremental_IWS_vs_ISS.py in the paper 
"AU-NN: ANFIS Unit Neural Network"

arXiv: 
"""



# To avoid innecesary log messages
from silence_tensorflow import silence_tensorflow
silence_tensorflow()


# Libraries needed
import tensorflow as tf
import numpy as np
import os
import json

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from AU_NN import AU_NN


def Extract_subject_folds(subject, script_dir):
    """
    Extract the 10 folds of the selected subject and dataset
    In each fold is included the train and test sets

    Parameters
    ----------
    subject : The subject we want the dataset
    script_dir : Tha path to the main code Incremental_IWS_vs_ISS.py

    Returns
    -------
    SubjectFolds : The folds of the subject
    """

    rel_path = 'DataSets/S' + str(subject + 1) + '.txt'
    file_path = os.path.join(script_dir, rel_path)
    with open(file_path) as f:
        SubjectFolds = json.load(f)
    return SubjectFolds



def BalanceCorpusMore(Corpus):
    """
    Takes an unbalanced corpus and outputs a corpus with balanced classes of 0 and 1

    Parameters
    ----------
    Corpus : The unbalanced corpus

    Returns
    -------
    BalancedCorpus : The corpus with balanced classes
    """

    # Creates an empy array of same lenght as features
    Instances_0 = np.zeros([Corpus.shape[1]])
    Instances_1 = np.zeros([Corpus.shape[1]])
    
    # Separates the 0 instances from the 1 instances
    for i in range(Corpus.shape[0]):
        if Corpus[i,-1] == 0:
            Instances_0 = np.vstack((Instances_0, Corpus[i,:]))
        else:
            Instances_1 = np.vstack((Instances_1, Corpus[i,:]))
    
    # Removes first row because is zero
    Instances_0 = np.delete(Instances_0, (0), axis=0)
    Instances_1 = np.delete(Instances_1, (0), axis=0)
    
    # Finds which instances set is bigger, if 0 or 1.
    # Then takes the smaller set and fills it with random repetitions from the same set
    # to match the same number as the bigger set
    while (Instances_0.shape[0] - Instances_1.shape[0]) != 0:
        if Instances_1.shape[0] < Instances_0.shape[0]:
            dif = Instances_0.shape[0] - Instances_1.shape[0]
            Instances_1 = np.vstack((Instances_1, Instances_1[:dif,:]))
        elif Instances_1.shape[0] > Instances_0.shape[0]:
            dif = Instances_1.shape[0] - Instances_0.shape[0]
            Instances_0 = np.vstack((Instances_0, Instances_0[0:dif,:]))
    BalancedCorpus = np.vstack((Instances_0, Instances_1))
    
    return BalancedCorpus


def Evaluate_subject(subject, clf, script_dir):
    """
    Function that takes a subject, trains its classifier and test them

    Parameters
    ----------
    subject : The subject we wish to evaluate
    clf : The AU-NN
    script_dir : the path o fhte main code Incremental_IWS_vs_ISS.py

    Returns
    -------
    Results : A dictionary that contains several metrics on the three 
            classifiers (incremental AU-NNN, full AU-NN, RF) of each fold in the subject

    """
      
    # Get the folds of the subject
    SubjectCorpus = Extract_subject_folds(subject, script_dir)
    Results = []

    # run for every fold (10)
    for fold in range(len(SubjectCorpus)):
        
        # Extract the fold
        CorpusFold = SubjectCorpus[fold]
        
        # Define the AU-NN
        Arch = clf['Arch']
        Learning_rates = clf['Learning_rates']
        epochs_1 = clf['epochs'][0]
        Clf = AU_NN(Arch)
        
        # Initialize results
        FoldResults = {}
        Incremental_results = []
    
        # Creates the scaler to noramlize features
        scaler = StandardScaler() 
        
        # From the selected fold, gets the train and test trials
        TrainSet = CorpusFold['TrainSet']
        TestSet = np.array(CorpusFold['TestSet'])
    
        # Run for every sesion to train (5)
        for session in range(len(TrainSet)):
            
            # Extract the train session, balance the classes, and randomize it
            TrainSession = np.array(TrainSet[session])
            BalancedTrainSession = BalanceCorpusMore(TrainSession)
            np.random.shuffle(BalancedTrainSession)
            
            # Extract the unnormalized X_train and y_train set
            Unnormalized_Xtrain = BalancedTrainSession[:,:-1]
            SessionTargets = BalancedTrainSession[:,-1]
            
            # Stacks new instances of new sessions to the X_train
            if session == 0:
                X_train = Unnormalized_Xtrain
                y_train = SessionTargets
                y_train = y_train.reshape(len(y_train),1) 
    
            else:
                X_train = np.vstack((X_train, Unnormalized_Xtrain))
                y_train = np.vstack((y_train, SessionTargets.reshape(len(BalancedTrainSession[:,-1]),1)  ))
    
            # Extract the X_test and y_set set
            X_test = TestSet[:,:-1]
            y_test = TestSet[:,-1]
            
            # Train the normalizer, and normalize the train and test instances
            scaler.fit(X_train) 
            X_train = scaler.transform(X_train) 
            X_test = scaler.transform(X_test)
           
            # Create the y_train in one_hot fashion
            depth = 2
            y_train_hot = tf.one_hot(y_train, depth, dtype=tf.float64)
            y_train_hot = tf.transpose(tf.reshape(y_train_hot[:,:,:],[y_train_hot.shape[0],depth]))
            y_train_numpy = y_train.ravel()
            
            # Initialize the metrics
            train_accuracy, test_accuracy, tn, fp, fn, tp, Specificity, Sensitivity, Tona_measure, f1score \
                = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            
            # Run for every learning rate in schedule (1)
            for learning_rate in Learning_rates:
                    
                # Train the incremental learning AU-NN
                Clf.fit(X_train,y_train_hot,epochs_1,learning_rate)
                       
            # Get the predictions on the train set
            train_predictions = Clf.predict(X_train)
            train_predictions = np.argmax(train_predictions,axis=0)
            train_pred_numpy = np.clip(train_predictions,0,1)
            
            # Get the predictions on the test set
            test_predictions = Clf.predict(X_test)
            test_predictions = np.argmax(test_predictions,axis=0)
            test_pred_numpy = np.clip(test_predictions,0,1)
            
            # Calculate the metrics for the session
            train_accuracy = accuracy_score(y_train_numpy, train_pred_numpy)
            test_accuracy = accuracy_score(y_test, test_pred_numpy)
            tn, fp, fn, tp = confusion_matrix(y_test, test_pred_numpy).ravel()
            Specificity = tn / (tn + fp)
            Sensitivity = tp / (tp + fn)
            Tona_measure = (Sensitivity + Specificity)/2
            f1score = f1_score(y_test, test_pred_numpy, average='binary')

            # Show the metrics of the session
            print('Cost function:', "%.4f" % Clf.cost_function(X_train,y_train_hot).numpy(), \
                  'Train accuracy:', "%.4f" % train_accuracy, \
                  'Test accuracy:', "%.4f" % test_accuracy, \
                  'Epoch:', epochs_1, ' Session:', (session + 1))
            print(confusion_matrix(y_test, test_pred_numpy))
            print('Specificity:', "%.4f" % Specificity, \
                  'Sensitivity', "%.4f" % Sensitivity, \
                  'Tona measure:', "%.4f" % Tona_measure, \
                  'f1 score:', "%.4f" % f1score )
            print('------------------------------')
                        
            # Store the results of the session
            Incremental_results.append( {'trn_acc': train_accuracy, 'tst_acc': test_accuracy,  \
                                 'specificity': Specificity, 'sensitivity': Sensitivity, \
                                 'tona_measure': Tona_measure, 'f1_score': f1score, \
                                 'tn': tn.astype('float64'), 'fp': fp.astype('float64'), \
                                 'fn': fn.astype('float64'), 'tp': tp.astype('float64')} )
                        
        # Store the results of the AU-NN of the fold
        FoldResults['AU_NN_incremental_results'] = Incremental_results
        print('------------------------------')
    
    
    
        # Train the Rf classifier with the full train set
        rf = RandomForestClassifier(max_depth=1,n_estimators=500, random_state=0)
        rf.fit(X_train, y_train_numpy)
        rf_train_pred = rf.predict(X_train)
        rf_test_pred = rf.predict( X_test)
        
        # Calculate the metrics for RF
        train_accuracy = accuracy_score(y_train_numpy, rf_train_pred)
        test_accuracy = accuracy_score(y_test, rf_test_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, rf_test_pred).ravel()
        Specificity = tn / (tn + fp)
        Sensitivity = tp / (tp + fn)
        Tona_measure = (Sensitivity + Specificity)/2
        f1score = f1_score(y_test, rf_test_pred, average='binary') 
        
        # Show the metrics of the RF
        print('RF Train accuracy:', "%.4f" % train_accuracy, \
                              'RF Test accuracy:', "%.4f" % test_accuracy )
        print(confusion_matrix(y_test, rf_test_pred))
        print('Specificity:', "%.4f" % Specificity, \
              'Sensitivity', "%.4f" % Sensitivity, \
              'Tona measure:', "%.4f" % Tona_measure, \
              'f1 score:', "%.4f" % f1score)
        print('------------------------------')
        
        # Store the results of the RF of the fold
        FoldResults['RF_results'] = {'trn_acc': train_accuracy, 'tst_acc': test_accuracy,  \
                                 'specificity': Specificity, 'sensitivity': Sensitivity, \
                                 'tona_measure': Tona_measure, 'f1_score': f1score, \
                                 'tn': tn.astype('float64'), 'fp': fp.astype('float64'), \
                                 'fn': fn.astype('float64'), 'tp': tp.astype('float64')}
            
        # Define the AU-NN to be trained with the full train set
        epochs_2 = clf['epochs'][1]
        AU_NN_2 = AU_NN(Arch)
        
        # Train for each learning rate on the schedule (1)
        for learning_rate in Learning_rates:
            
                # Train the full AU-NN
                AU_NN_2.fit(X_train,y_train_hot,epochs_2,learning_rate)    
                        
                # Get the predictions on the train set
                train_predictions = AU_NN_2.predict(X_train)
                train_predictions = np.argmax(train_predictions,axis=0)
                train_pred_numpy = np.clip(train_predictions,0,1)
                
                # Get the predictions on the test set
                test_predictions = AU_NN_2.predict(X_test)
                test_predictions = np.argmax(test_predictions,axis=0)
                test_pred_numpy = np.clip(test_predictions,0,1)
                
                # Calculate the metrics for the full AU-NN
                train_accuracy = accuracy_score(y_train_numpy, train_pred_numpy)
                test_accuracy = accuracy_score(y_test, test_pred_numpy)
                tn, fp, fn, tp = confusion_matrix(y_test, test_pred_numpy).ravel()
                Specificity = tn / (tn + fp)
                Sensitivity = tp / (tp + fn)
                Tona_measure = (Sensitivity + Specificity)/2
                f1score = f1_score(y_test, test_pred_numpy, average='binary')

                # Show the metrics of the full AU-NN
                print('Cost function:', "%.4f" % AU_NN_2.cost_function(X_train,y_train_hot).numpy(), \
                      'Train accuracy:', "%.4f" % train_accuracy, \
                      'Test accuracy:', "%.4f" % test_accuracy, \
                      'Epoch:', epochs_2)
                print(confusion_matrix(y_test, test_pred_numpy))
                print('Specificity:', "%.4f" % Specificity, \
                      'Sensitivity', "%.4f" % Sensitivity, \
                      'Tona measure:', "%.4f" % Tona_measure, \
                      'f1 score:', "%.4f" % f1score )
        print('------------------------------')
        
        # Store the results of the full Au-NN of the fold
        FoldResults['AU_NN_full_results'] = {'trn_acc': train_accuracy, 'tst_acc': test_accuracy,  \
                                 'specificity': Specificity, 'sensitivity': Sensitivity, \
                                 'tona_measure': Tona_measure, 'f1_score': f1score, \
                                 'tn': tn.astype('float64'), 'fp': fp.astype('float64'), \
                                 'fn': fn.astype('float64'), 'tp': tp.astype('float64')}
            
        # Store the results of the three classifiers for the fold
        Results.append(FoldResults)  
        
    return Results



