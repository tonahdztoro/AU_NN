# -*- coding: utf-8 -*-
"""
last update 15/03/22
@author: Tonatiuh Hernández-Del-Toro (tonahdztoro@gmail.com)

Code to plot the results obtained in the classification of imagined words using the AU-NN in the paper 
"AU-NN: ANFIS Unit Neural Network"

arXiv: 
"""

# Libraries needed
import matplotlib.pyplot as plt
import numpy as np


# Load the obtained results with the code IW_classification.py
subject_accuracies = np.zeros((1,5))
for subject in range(27):
    accuracies = np.load('Results/S' + str(subject + 1) + 'accuracies.npy')
    subject_accuracies = np.vstack((subject_accuracies, accuracies))
subject_accuracies = subject_accuracies[1:,:]

# Calculate the mean of the folds, and add the mean over all the subjects
mean_accuracy = np.mean(subject_accuracies,axis=1)
mean_accuracy = np.append(mean_accuracy, np.mean(mean_accuracy))
mean_accuracy = mean_accuracy.tolist()

# Calculate the standard deviation of the folds, and add the standard deviation over all the subjects
std_accuracy = np.std(subject_accuracies,axis=1)
std_accuracy = np.append(std_accuracy, np.std(mean_accuracy))
std_accuracy = std_accuracy.tolist()


# Results reported in Torres-Garcia2016
Torres_Garcia_benchmark = [0.9026, 0.5639, 0.7467, 0.7709, 0.7573, 0.4242, 0.725 , 0.9147,
        0.7639, 0.6923, 0.914 , 0.8125, 0.7276, 0.5151, 0.8853, 0.6779,
        0.7518, 0.7985, 0.4919, 0.897 , 0.5154, 0.7095, 0.6923, 0.5691,
        0.3106, 0.6617, 0.797 , 0.7033]
Torres_Garcia_std = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.std(Torres_Garcia_benchmark)]


# Results Reported in Garcia-Salinas2019
Garcia_Salinas_benchmark = [0.6103, 0.8059, 0.764 , 0.8261, 0.8412, 0.7382, 0.7504, 0.6724,
        0.8   , 0.782 , 0.661 , 0.7103, 0.7132, 0.7408, 0.8423, 0.7603,
        0.711 , 0.8243, 0.7588, 0.8246, 0.8132, 0.7504, 0.7643, 0.7813,
        0.5941, 0.679 , 0.7938, 0.759 ]
Garcia_Salinas_std = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.std(Garcia_Salinas_benchmark)]



# Plot the results
x = np.arange(1,29)
width = 0.3

fig, ax = plt.subplots(figsize=(15,8))
fig.tight_layout()

AU_NN = ax.bar(x - width, mean_accuracy, yerr=std_accuracy, alpha=0.8, capsize=4, color = 'tab:blue', width = width)
Torres_Garcia = ax.bar(x, Torres_Garcia_benchmark, yerr=Torres_Garcia_std, capsize=4,  color = 'tab:orange', alpha=0.8, width = width)
Garcia_Salinas = ax.bar(x + width, Garcia_Salinas_benchmark, yerr=Garcia_Salinas_std, capsize=4,  color = 'tab:green', alpha=0.8, width = width)

ax.set_ylabel('Accuracy')
ax.set_xlabel('Subjects')
ax.set_title('Accuracy per subject')
ax.set_xticks(x)
ax.grid(axis='y')
plt.ylim(0, 1.0)
plt.yticks(np.arange(0, 1.1, 0.1))
ax.legend()

plt.show()

# Save the figure
fig.savefig('Plots/results.pdf', dpi=20, bbox_inches="tight")