# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 22:33:49 2023

@author: elie1
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Create list with all files from the same subtype
def createFileList(subtype):
    filesList = []
    #for each file
    for file in os.listdir():
        #if contains string
        if subtype in file and os.path.isfile(file) :
            filesList.append(file)
    return filesList

#List of files
fileList_subtype01 = createFileList("subtype01")
fileList_subtype02 = createFileList("subtype02")

def organize_data(fileList):
    #result dataframe
    results = pd.DataFrame(columns=["Differenciation", "Reduction"])
    
    #for each file
    for file in fileList :
        
        #read file
        raw_data = pd.read_csv(file, header = 0, index_col = 0) 
        
        #Rate of SIB per 1,000 generation in each condition
        alone_SIB = np.sum(np.logical_and
                            (raw_data['Behavior'][0:10000] > 470,
                             raw_data['Behavior'][0:10000] < 512))/10 
        
        play_SIB = np.sum(np.logical_and
                            (raw_data['Behavior'][10000:20000] > 470,
                             raw_data['Behavior'][10000:20000] < 512))/10 
        
        treatment_SIB = np.sum(np.logical_and
                            (raw_data['Behavior'][20000:30000] > 470,
                             raw_data['Behavior'][20000:30000] < 512))/10 
        
        
        #percentage of differenciation
        percentage_differenciation = (1-(play_SIB/alone_SIB))*100
        
        #percentage of reduction
        percentage_reduction = (1 - (treatment_SIB/alone_SIB))*100
                                                                        
        #append to results
        results = results.append({"Differenciation": percentage_differenciation,
                                  "Reduction": percentage_reduction},
                                 ignore_index=True)
    return results


#create results data frame for subtype 1 and 2
results_subtype01 = organize_data(fileList_subtype01)
results_subtype02 = organize_data(fileList_subtype02)


def scatter_plot(df1, df2):
         
    plt.scatter(x="Differenciation", y="Reduction", data=df1, color="black", 
                    label='Subtype 1')
    
    plt.scatter(x="Differenciation", y="Reduction", data=df2, color="red", 
                    label='Subtype 2')
  
    #labels and title
    plt.xlabel("Percentage Differenciation")
    plt.ylabel("Percentage Reduction")
    plt.title("Morris & Lucia (2023) models")
    
    
    plt.axis( [-25,100, -25, 110])
    plt.xticks([-25, 0, 25, 50, 75, 100])
    plt.yticks([-25, 0, 25, 50, 75, 100])
    
    #legend
    plt.legend()

    #show plot
    plt.show()


scatter_plot(results_subtype01, results_subtype02)
