# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 13:44:45 2023

@author: elie1
"""
#Import functions and packages
import numpy as np
import pandas as pd
import time
from joblib import Parallel, delayed

#Initialize population (size: number of behaviors in population)
def initialize_population(size = 100):
    #Randomly select population of the given size
    return np.random.choice(range(1024), size, replace = False)

#Initialize backgroud class (behavior_class: class to exclude from sampling):
def initialize_background(size, behavior_class):
    sampling = np.array(range(1024))
    sampling = np.delete(sampling, 
                         list(range(behavior_class[0], behavior_class[1]+1)))
    return np.random.choice(sampling, size, replace = False)
    
#Select behavior (population: population of behaviors)
def select_behavior(population):
    #Randomly select one behavior from population
    return  np.random.choice(population, 1)

#Check if behavior belongs to a behavior class 
def class_check(behavior, behavior_class):
    #Check if behavior is in class return True
    return behavior in range(behavior_class[0], behavior_class[1]+1)
        
#Check if behavior belongs to background
def background_check(behavior, background_behaviors):
    return behavior in background_behaviors

#Compute fitness
def compute_fitness(population, behavior_class):
    #If the behavior class is a range
    if len(behavior_class) > 1:
        #Get midpoint
        midpoint = int(np.median(behavior_class))   
    #If the behavior_class is a specific value (background reinforcement)
    else: 
        midpoint = int(behavior_class)   
    #For each behavior in population compute fitness using the absolute diff
    fitness_list = []
    for behavior in population:
        fitness_list.append(np.abs(behavior - midpoint))  
    return np.array(fitness_list).flatten()   

#Select fitness value from linear function
def select_fitness_value(mean_fitness):
    fitness_value = 3*mean_fitness*(1- np.sqrt(1-np.random.random()))
    return np.int32(fitness_value)
    
#Select two parents by fitness
def select_parents_by_fitness(population, behavior_class, mean_fitness):
    #Compute fitness
    fitness_list = compute_fitness(population, behavior_class)  
    
    #No father
    father_identified = False
    
    #Continue until father is identifieds
    while father_identified == False:
        #Select fitness value for father
        fitness_value_father = select_fitness_value(mean_fitness)
        #Identify which the index of the value
        idx_father, = np.where(fitness_list == fitness_value_father)
        
        #If one matching fitness value in population
        if idx_father.size == 1:
            father = population[idx_father]
            father_identified = True
        #If more than one matching fitness in population
        if idx_father.size > 1:
            idx_father = np.random.choice(idx_father, 1)
            father = population[idx_father]
            father_identified = True
       
    #No mother
    mother_identified = False
    
    '''
    #If all 0 probability except father, randomly pick mother
    if np.sum(fitness_list >= 3*mean_fitness) == 99:
        idx, = np.where(fitness_list >= 3*mean_fitness)
        idx = np.random.choice(idx, 1)
        mother = population[idx]
        mother_identified = True
    '''
    count = 0
    #Continue until mother is identified
    while mother_identified == False:
        
        count = count+1
        #Select fitness value for mother
        fitness_value_mother = select_fitness_value(mean_fitness)
        
        #If fitness value is different from father
        if fitness_value_mother != fitness_value_father:
            #Identify which the index of the value
            idx_mother, = np.where(fitness_list == fitness_value_mother)
            
            #If one matching fitness value in population
            if idx_mother.size == 1:
                mother = population[idx_mother]
                mother_identified = True
                
            #If more than one matching fitness in population
            if idx_mother.size > 1:
                idx_mother = np.random.choice(idx_mother, 1)
                mother = population[idx_mother]
                mother_identified = True
        
        if count > 1000:
            idx_choices = np.delete(np.array(range(100)), idx_father)
            idx_mother = np.random.choice(idx_choices, 1)
            mother = population[idx_mother]
            mother_identified = True
        
    return father, mother

#Randomly select two parents
def select_parents_randomly(population):
    #Randomly select parents
    parents = np.random.choice(population, 2, replace = False)
    #Return each parent
    return parents[0], parents[1]

#Reproduction (bitwise)
def reproduce(father, mother):   
    #Tranform parent value from decimal to binary
    bin_father = np.binary_repr(int(father), 10)
    bin_mother = np.binary_repr(int(mother), 10)       
    #Create string for individual child
    child = str()
    #Bitwise selection of each of the 10 bits (0.5 probability)
    for j in range(10):
        #Randomly select father or mother
        inheritor =  np.random.choice(('mother', 'father'))
        #If inherits from father 
        if inheritor == 'father':
            child = child + bin_father[j]
        #If inherits from mother
        else:
            child = child + bin_mother[j]
    return child

#Produce new_population
def produce_children(population, behavior_class = 0, mean_fitness = 0, 
                     random = False):    
    #Create children list
    children = [] 
    #Produce a number of children equal to population length
    for i in range(len(population)):
        #If parents are randomly selected
        if random == True:
            father, mother = select_parents_randomly(population)
        
        #If parents are selected by fitness
        else: 
            father, mother = select_parents_by_fitness(population, 
                                                       behavior_class, 
                                                       mean_fitness)
        #Produce child
        child = reproduce(father, mother)
        
        #Add to list of children
        children.append(child)
    return children 

#Bitflip
def bitflip(child):   
    #Select bit to flip
    bit_idx = np.random.choice(10)   
    #Transform to list
    child = list(child)
    #Replace bit of 0 by 1 
    if child[bit_idx] == '0':       
        child[bit_idx] = '1'       
        #Transform back to string
        child = "".join(child)   
    #Replace bit of 1 by 0 
    else:
        child[bit_idx] = '0'       
        #Transform back to string
        child = "".join(child)   
    return child

#Mutation (bitflip)
def mutate_children(children, mutation_rate):   
    
    #Exception for .005
    if mutation_rate == 0.005:
        mutation_rate = np.random.choice([0, 0.01])
     
    try:
        #Randomly select children indices to mutate
        children_idx = np.random.choice(len(children), replace = False,
                                    size = int(mutation_rate*len(children)))   
        #For each child to mutate
        for child_idx in children_idx:       
            #Extract child
            child = children[child_idx]       
            #Replace child by child with bitflip
            children[child_idx] = bitflip(child)            
        ##Convert back to decimal
        #Create new vector
        new_population = np.empty((0,), dtype= np.int32) 
        #For each child
        for i in range(len(children)):
            #Add decimal value to vector
            new_population = np.hstack((new_population, int(children[i],2)))
        
    except:
        new_population = children.copy()
            
    return new_population

#check schedule (interval: interval, current_value: schedule on or off)
def check_schedule(interval, current_value):   
    #If schedule is on, keep on 
    if current_value == 1:
        return 1    
    #Else return new schedule value based on interval probability
    else:
        #Set the probability
        prob = 1/interval
        #Check the probability
        return np.random.choice(2, 1, p = [1-prob, prob])

#Functional analysis
def FA_and_treatment(AO, generations) :
    
    #Target label
    target_class = (471, 511)
    
    #Set background for all conditions
    background_behaviors = initialize_background(200, target_class) 
    
    #set interval
    target_interval = 20
    background_interval = 20
    
    #for each subtype
    for subtype in range(1,3):
        #set mutation rate and mf for subtype 1
        if subtype == 1 :
            
            mutation_rate = 0.005
            target_mf = 100
        
        #set mutation rate and mf for subtype 2
        if subtype == 2 :
            
            mutation_rate = 0.3
            target_mf = 50
        
        #results dataframe
        results = pd.DataFrame(columns=['Condition', 'Time', 'Behavior', 
                                        'Target_Reinf', 'Alt_Reinf'])
    
        #for each condition
        for condition in range(1,4):
            #set mf for alone condition
            if condition == 1 :
                background_mf = 200
                condition_name = "Alone"
            
            #set mf for play condition
            if condition == 2 :
                background_mf = 50
                condition_name = "Play"
            
            #set mf for treatment
            if condition == 3 :
                background_mf = 50
                condition_name = "Treatment"
            
            #Is the schedule on or off
            target_sched = 0
            background_sched = 0
            
            #Reinitialize population for each condition
            population = initialize_population(100)
                 
            for i in range(generations) :
                #Select behavior from population
                print(i)
                behavior = select_behavior(population)
            
                #Check if behavior belongs to class
                is_target = class_check(behavior, target_class)
                is_background = background_check(behavior, 
                                                 background_behaviors)
            
                #Check and update the schedules
                target_sched = check_schedule(target_interval, target_sched)
                background_sched = check_schedule(background_interval, 
                                                  background_sched)
                
                #Check if treatment condition running
                if condition == 3:
                    
                    #reinforce target behavior and produce children by fitness
                    if np.logical_and(is_target == True, target_sched == 1):
                        target_reinforced = True
                        background_reinforced = False
                        children = produce_children(population, target_class, 
                                                target_mf)
                        target_sched = 0
                    
                    #Continuous reinforcement of background
                    elif is_background == True:
                        target_reinforced = False
                        background_reinforced = True
                        children = produce_children(population, behavior, 
                                                    background_mf)
                    
                    #No reinforcement
                    else:
                        target_reinforced = False
                        background_reinforced = False
                        children = produce_children(population, random = True)
            
            
            
                #Reinforce target behavior and produce children by fitness
                elif np.logical_and(is_target == True, target_sched == 1):
                    target_reinforced = True
                    background_reinforced = False
                    children = produce_children(population, target_class, 
                                                target_mf)
                    target_sched = 0
        
                #Reinforce background behavior and select children by fitness
                elif np.logical_and(is_background == True, background_sched == 1):
                    background_reinforced = True
                    target_reinforced = False
                    children = produce_children(population, behavior, 
                                                background_mf)
                    background_sched = 0
        
                #No reinforcement and select children randomly
                else:
                    target_reinforced = False
                    background_reinforced = False
                    children = produce_children(population, random = True)
                
            
                #Mutation
                population = mutate_children(children, mutation_rate)
            
                #Add to results
                results.loc[len(results)] = [condition_name, i, int(behavior), 
                                    target_reinforced, background_reinforced]
                                                                           
        #send results to csv
        results.to_csv('AO'+ str("{:02d}".format(AO))  + "subtype" + 
                       str("{:02d}".format(subtype)) + ".csv")
        
        
#Parallelize
time_start = time.time()

Parallel(n_jobs=10)(delayed(FA_and_treatment)(AO = i + 1, generations = 10000, 
                                         ) for i in range(30)) 
print(time.time()-time_start)