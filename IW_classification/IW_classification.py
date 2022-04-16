# -*- coding: utf-8 -*-
"""
last update 15/03/22
@author: Tonatiuh Hernández-del-Toro (tonahdztoro@gmail.com)

Code to evaluate the AU-NN with the classification of imagined words in the paper 
"AU-NN: ANFIS Unit Neural Network"

arXiv: 
"""

# Libraries needed
import numpy as np
import IW_classification_functions as func
from AU_NN import AU_NN





if __name__ == '__main__':
    
    cut = 24 # To take 80% of words as train set and 20% of words as test set
    Folds = 5 # To evaluate 5 times in order to have better statistics
    
    # Run over every subject
    subjects = list(range(0,27))
    for subject in subjects:
        
        # Initialize arrays to strore results values
        subject_results = {}
        accuracies = np.array([])
        
        # Run over every fold
        for _ in range(Folds):
            
            # Create the train and test set of the fold
            X_train, y_train_hot, X_y_test = func.create_fold(subject,cut)
            
            # Define the model
            Arch = {'NoF': [70],
                    'NoMF': [70],
                    'NoAU_Layers': [5]}
            model = AU_NN(Arch)
            
            # Train and evaluate the model with the obtained train and test sets
            fold_accuracy = func.evaluate_subject_fold(X_train, y_train_hot, X_y_test, model)
            
            # Store the results obtained in the fold
            accuracies = np.append(accuracies, fold_accuracy)    
                    
        # Store the results obtained for each subject
        subject_results['fold_accuracies'] = accuracies
        subject_results['mean'] = np.mean(accuracies)
        subject_results['std'] = np.std(accuracies)
    
        # Save the results
        np.save('Results/S' + str(subject + 1) + 'accuracies.npy', accuracies)
    
    
    
    
    
    



