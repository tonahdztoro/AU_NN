# -*- coding: utf-8 -*-
"""
last update 15/03/22
@author: Tonatiuh Hernández-Del-Toro (tonahdztoro@gmail.com)

Code to evaluate the AU-NN with the incremental learning IWS vs. ISS in the paper 
"AU-NN: ANFIS Unit Neural Network"

arXiv: 
"""


import Incremental_IWS_vs_ISS_functions as func
import json
import os



# Set the directory to save the results
script_dir = os.path.dirname(__file__)
results_path = 'Results/'
file_path = os.path.join(script_dir, results_path)



if __name__ == '__main__':
    

    # Define the AU-NN and training hyperparameters
    Learning_rates = [0.001]
    epochs = [300, 1200]
    Arch = {'NoF': [70,10],
        'NoMF': [3,3],
        'NoAU_Layers': [10,2]}
    clf = {'Arch' : Arch, 'Learning_rates' : Learning_rates, 'epochs' : epochs}
    

    subjects = list(range(0,27))
    for subject in subjects: # Through each subject
    
        # Evaluate the AU-NN with the subject's folds
        Results = func.Evaluate_subject(subject, clf, script_dir)
        
        # Save the results
        temp_file = open(file_path + 'incremental_results_S' + str(subject + 1), "w")
        json.dump(Results, temp_file)
        temp_file.close()
        print('I just finished subject', (subject + 1) )
        
        
      
    print('I finished everything')