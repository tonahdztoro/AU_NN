# -*- coding: utf-8 -*-
"""
last update 15/03/22
@author: Tonatiuh Hernández-Del-Toro (tonahdztoro@gmail.com)

Code to plot the results obtained in the incremental learning sessions using the AU-NN in the paper 
"AU-NN: ANFIS Unit Neural Network"

arXiv: 
"""

# Import libraries needed
import json
import os
import numpy as np
from matplotlib import pyplot as plt

def create_statistics(Results, metric):
    
    ReducedResults = {}
    ReducedResults['AU_NN_inc'] = {}
    FoldResults = Results
    
    AU_NN_full = []
    RF = []
    
    ReducedResults['AU_NN_inc'][0] = []
    ReducedResults['AU_NN_inc'][1] = []
    
    AU_NN_inc = np.zeros((10,5))
    
    for fold in range(len(FoldResults)):
        AU_NN_full_results = Results[fold]['AU_NN_full_results']
        RF_results = Results[fold]['RF_results']
        
        AU_NN_full.append( AU_NN_full_results[metric])
        RF.append( RF_results[metric])
        
        AU_NN_inc_results = Results[fold]['AU_NN_incremental_results']
        for session in range(len(AU_NN_inc_results)):
            AU_NN_inc_session = AU_NN_inc_results[session]
            AU_NN_inc[fold,session] = AU_NN_inc_session[metric]
    
    ReducedResults['AU_NN_inc'][0].append(np.mean(AU_NN_inc, axis=0))
    ReducedResults['AU_NN_inc'][1].append(np.std(AU_NN_inc, axis=0))
    
    ReducedResults['AU_NN_full'] = [np.mean(AU_NN_full), np.std(AU_NN_full)]
    ReducedResults['RF'] = [np.mean(RF), np.std(RF)]
    
    return ReducedResults

def create_full_statistics(metric):
    ReducedResults = {}
    ReducedResults['AU_NN_inc'] = {}
    
    AU_NN_full = []
    RF = []
    
    ReducedResults['AU_NN_inc'][0] = []
    ReducedResults['AU_NN_inc'][1] = []
    
    AU_NN_inc = np.zeros((270,5))
    
    
    for subject in range(0,27):
        
        # Set the directory to load the results
        script_dir = os.path.dirname(__file__)
        results_path = 'Results/'
        file_path = os.path.join(script_dir, results_path)
        temp_file = open(file_path + 'incremental_results_S' + str(subject + 1), "r")
        Results = json.loads(temp_file.read())
        FoldResults = Results
    
        for fold in range(len(FoldResults)):
            AU_NN_full_results = Results[fold]['AU_NN_full_results']
            RF_results = Results[fold]['RF_results']
            
            AU_NN_full.append( AU_NN_full_results[metric])
            RF.append( RF_results[metric])
            
            AU_NN_inc_results = Results[fold]['AU_NN_incremental_results']
            for session in range(len(AU_NN_inc_results)):
                AU_NN_inc_session = AU_NN_inc_results[session]
                AU_NN_inc[fold + (subject*10),session] = AU_NN_inc_session[metric]
        
    ReducedResults['AU_NN_inc'][0].append(np.mean(AU_NN_inc, axis=0))
    ReducedResults['AU_NN_inc'][1].append(np.std(AU_NN_inc, axis=0))
        
    ReducedResults['AU_NN_full'] = [np.mean(AU_NN_full), np.std(AU_NN_full)]
    ReducedResults['RF'] = [np.mean(RF), np.std(RF)]
    
    return ReducedResults


#%% Plot for every 27 subject
for subject in range(10,27):
    
    # Set the directory to load the results
    script_dir = os.path.dirname(__file__)
    results_path = 'Results/'
    file_path = os.path.join(script_dir, results_path)
    temp_file = open(file_path + 'incremental_results_S' + str(subject + 1), "r")
    Results = json.loads(temp_file.read())
    
    metrics = ['trn_acc', 'tst_acc', 'f1_score', 'sensitivity', 'specificity', 'tona_measure']

        
    # Plot
    n_rows = 3
    n_cols = 2
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True , figsize=(10,10))
    fig.text(0.5, 0.08, 'Sessions', ha='center')
    fig.text(0.35, 0.9, 'Different metrics for sessions for subject ' + str(subject + 1), ha='center')
    counter = 0
    
    for i in range(n_rows):
        for j in range(n_cols):
            
            ReducedResults = create_statistics(Results,metrics[counter])
            counter += 1
            
            # Data for plotting
            t = np.arange(1.0, 6.0, 1.0)
            AUNN_inc = np.array(ReducedResults['AU_NN_inc'][0]).reshape(5)
            AUNN_full = np.ones((5))*ReducedResults['AU_NN_full'][0]
            RF = np.ones((5))*ReducedResults['RF'][0]
            
            AUNN_inc_error = np.array(ReducedResults['AU_NN_inc'][1]).reshape(5)
            AUNN_full_error = np.ones((5))*ReducedResults['AU_NN_full'][1]
            RF_error = np.ones((5))*ReducedResults['RF'][1]
            
            
            axs[i,j].yaxis.grid(True)
            axs[i,j].set_ylim([0.0, 1.0])
            axs[i,j].set_xticks(np.arange(1, 6, 1))
    
            axs[i,j].plot(t, AUNN_inc, label='AU-NN inc', color='blue', linewidth=1, alpha=1.0)
            axs[i,j].fill_between(t, AUNN_inc - AUNN_inc_error, AUNN_inc + AUNN_inc_error, color='blue', linewidth=1, alpha=0.2)
            
            axs[i,j].plot(t, AUNN_full, label='AU-NN full', color='orange', linewidth=1, alpha=1.0)
            axs[i,j].fill_between(t, AUNN_full - AUNN_full_error, AUNN_full + AUNN_full_error, color='orange', linewidth=1, alpha=0.2)
            
            axs[i,j].plot(t, RF, label='RF', color='green', linewidth=1, alpha=1.0)
            axs[i,j].fill_between(t, RF - RF_error, RF + RF_error, color='green', linewidth=1, alpha=0.2)
            
    k = 0
    ylabels = ['Train accuracy', 'Test accuracy', 'F1 score', 'Sensitivity', 'Specificity', '(Sensitivity + Specificity)2']
    for ax in axs.flat:
            ax.set(ylabel=ylabels[k])
            k += 1
    
    

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    
    ax.legend(ncol=3, bbox_to_anchor=(1,3.6))
    
    
    plt.show()

    # Set the directory to save the plots
    script_dir = os.path.dirname(__file__)
    plots_path = 'Plots/'
    file_plots_path = os.path.join(script_dir, plots_path)
    fig.savefig(file_plots_path + 'S' + str(subject + 1) + '.pdf', dpi=10, bbox_inches="tight")






#%% Plot for last subject (average)



metrics = ['trn_acc', 'tst_acc', 'f1_score', 'sensitivity', 'specificity', 'tona_measure']
ReducedResults = {}

# Plot
n_rows = 3
n_cols = 2
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True , figsize=(10,10))
fig.text(0.5, 0.08, 'Sessions', ha='center')
fig.text(0.35, 0.9, 'Average metrics for sessions for all subjects', ha='center')
counter = 0

for i in range(n_rows):
    for j in range(n_cols):
        
        ReducedResults[metrics[counter]] = create_full_statistics(metrics[counter])
        ResultsToPlot = ReducedResults[metrics[counter]]
        counter += 1
        
        # Data for plotting
        t = np.arange(1.0, 6.0, 1.0)
        AUNN_inc = np.array(ResultsToPlot['AU_NN_inc'][0]).reshape(5)
        AUNN_full = np.ones((5))*ResultsToPlot['AU_NN_full'][0]
        RF = np.ones((5))*ResultsToPlot['RF'][0]
        
        AUNN_inc_error = np.array(ResultsToPlot['AU_NN_inc'][1]).reshape(5)
        AUNN_full_error = np.ones((5))*ResultsToPlot['AU_NN_full'][1]
        RF_error = np.ones((5))*ResultsToPlot['RF'][1]
        
        
        axs[i,j].yaxis.grid(True)
        axs[i,j].set_ylim([0.0, 1.0])
        axs[i,j].set_xticks(np.arange(1, 6, 1))

        axs[i,j].plot(t, AUNN_inc, label='AU-NN inc', color='blue', linewidth=1, alpha=1.0)
        axs[i,j].fill_between(t, AUNN_inc - AUNN_inc_error, AUNN_inc + AUNN_inc_error, color='blue', linewidth=1, alpha=0.2)
        
        axs[i,j].plot(t, AUNN_full, label='AU-NN full', color='orange', linewidth=1, alpha=1.0)
        axs[i,j].fill_between(t, AUNN_full - AUNN_full_error, AUNN_full + AUNN_full_error, color='orange', linewidth=1, alpha=0.2)
        
        axs[i,j].plot(t, RF, label='RF', color='green', linewidth=1, alpha=1.0)
        axs[i,j].fill_between(t, RF - RF_error, RF + RF_error, color='green', linewidth=1, alpha=0.2)
        
k = 0
ylabels = ['Train accuracy', 'Test accuracy', 'F1 score', 'Sensitivity', 'Specificity', '(Sensitivity + Specificity)2']
for ax in axs.flat:
        ax.set(ylabel=ylabels[k])
        k += 1


lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

ax.legend(ncol=3, bbox_to_anchor=(1,3.6))


plt.show()

# Set the directory to save the plot
script_dir = os.path.dirname(__file__)
plots_path = 'Plots/'
file_plots_path = os.path.join(script_dir, plots_path)
fig.savefig(file_plots_path + 'S' + str(28) + '.pdf', dpi=10, bbox_inches="tight")





