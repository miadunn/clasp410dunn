#!/usr/bin/env python3

'''
This file contains tools and scripts for completing Lab 1 for CLaSP 410.
To reproduce the plots shown in the lab report, do this...

This file performs fire/disease spread simulations. 
To get solution for lab 1: run these commands:
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

forest_cmap = ListedColormap(['tan', 'darkgreen', 'crimson'])

def fire_spread(nNorth=3, nEast=3, maxiter=4, pspread=1.0, pbare=0.0, pstart=0.0):
    '''
    This function performs a fire/disease spread simulation.

    Parameters
    ==========
    nNorth, nEast : integer, defaults to 3
        Set the north-south (i) and east-west (j) size of grid
        Default is 3 squares in each direction
    maxiter : int, defaults to 4
        Set the maximum number of iterations including initial condition
    pspread: float, defaults to 1
        chance for fire to spread
    pbare: float, defaults to 0
        change for cells to start as a bare patch
    pstart: float, defaults to 0
        chance for cells to start on fire
    '''

    # Create forest and set initial conditions
    forest = np.zeros([maxiter, nNorth, nEast]) + 2

    # Set fire! To the center of the forest.
    istart, jstart = nNorth//2, nEast//2
    forest[0, istart, jstart] = 3

    # determine if cells are bare to start
    for i in range(nNorth):
        for j in range(nEast):
            # roll dice to see if it's a bare spot
            if np.random.rand() < pbare:
                forest[0, i, j] = 1
    #print(forest)            

    # Plot initial condition
    fig, ax = plt.subplots(1, 1)
    contour = ax.matshow(forest[0, :, :],cmap=forest_cmap, vmin=1, vmax=3)
    ax.set_title(f'Iteration = {0:03d}')
    plt.colorbar(contour, ax=ax)
    fig.savefig('inital_conditions.png')

    # Propagate the solution
    for k in range(maxiter-1):
        #set change to burn
        ignite = np.random.rand(nNorth, nEast)

        #use current step to set next step:
        forest[k+1, :, :] = forest[k, :, :]

        # burn from north to south
        doburn = (forest[k, :-1, :] == 3) & (forest[k, 1:, :] == 2) & \
            (ignite[1:, :] <= pspread)
        forest[k+1, 1:, :][doburn] = 3

        #Burn from south to north.
        doburn = (forest[k, :-1, :] == 2) & (forest[k, 1:, :] == 3) & \
            (ignite[:-1, :] <= pspread)
        forest[k+1, :-1, :][doburn] = 3
               

        #Burn from west to east.
        doburn = (forest[k, :, 1:] == 2) & (forest[k, :, :-1] == 3) & \
            (ignite[:, 1:] <= pspread)
        forest[k+1, :, 1:][doburn] = 3

        #Burn in each cardinal direction from east to west.
        doburn = (forest[k, :, :-1] == 2) & (forest[k, :, 1:] == 3) & \
            (ignite[:, :-1] <= pspread)
        forest[k+1, :, :-1][doburn] = 3
            
        
        #set currently burning to bare
        wasburn = forest[k, :, :] == 3 # find cells that WERE burning
        forest[k+1, wasburn] = 1     # ... they are now bare

        # plot initial condition
        fig, ax = plt.subplots(1,1)
        contour = ax.matshow(forest[k+1, :, :], cmap=forest_cmap, vmin=1, vmax=3)
        ax.set_title(f'Iteration = {k+1:03d}')
        plt.colorbar(contour, ax=ax)

        fig.savefig(f'fig{k:04d}.png')
        plt.close('all')

        nBurn = (forest[k+1, :,:] == 3).sum()
        if nBurn == 0:
            print(f"burn completed in {k+1} steps")
            return k+1

# define burn rate
def explore_burnrate():
    ''' Vary burn rate and see how fast fire ends.'''

    # set probability of burning the next cell from 0 to 1 with a step of 0.05
    prob = np.arange(0, 1.05, .05)
    nsteps = np.zeros(prob.size)

    # for each probability of spread, print number of steps to complete
    for i, p in enumerate(prob):
        # if statement doesn't print: maxiter isnt big enough
        print(f"Burning for pspread = {p}")
        nsteps[i] = fire_spread(nNorth=100, nEast=100, pspread=p, maxiter=400)

    plt.plot(prob, nsteps)
    plt.savefig('prob_nstep_pspread.png')

# vary bare cells
def vary_bare():
    ''' Vary amount of initial bare cells to see how fire spreads'''
    # set probability of cell starting as bare
    prob = np.arange(0, 1.05, .05)
    nsteps = np.zeros(prob.size)

    for i, p in enumerate(prob):
        print(f"Burning of pbare = {p}")
        nsteps[i] = fire_spread(nNorth=100, nEast=100, pbare=p, maxiter=400)
    
    plt.plot(prob, nsteps)
    plt.savefig('prob_nstep_pbare.png')


disease_cmap = ListedColormap(['black', 'blue', 'darkgreen', 'crimson'])

# disease
def disease_spread(nNorth=100, nEast=100, maxiter=400, pspread=1.0, pbare=0.2, pstart=0.0, pfatal=0.2):
    '''
    This function performs a fire/disease spread simulation.

    Parameters
    ==========
    nNorth, nEast : integer, defaults to 3
        Set the north-south (i) and east-west (j) size of grid
        Default is 3 squares in each direction
    maxiter : int, defaults to 4
        Set the maximum number of iterations including initial condition
    pspread: float, defaults to 1
        chance for disease to spread
    pbare: float, defaults to 0
        change for peaople to start as immune
    pstart: float, defaults to 0
        chance for people to start as sick
    pfatal: float, defaults to 0
        chance for sick person to die
    '''

    # Create forest and set initial conditions
    people = np.zeros([maxiter, nNorth, nEast]) + 2

    # Set fire! To the center of the forest.
    istart, jstart = nNorth//2, nEast//2
    people[0, istart, jstart] = 3

    # determine if cells are bare to start
    for i in range(nNorth):
        for j in range(nEast):
            # roll dice to see if it's a bare spot
            if np.random.rand() < pbare:
                people[0, i, j] = 1
    #print(forest)            

    # Plot initial condition
    fig, ax = plt.subplots(1, 1)
    contour = ax.matshow(people[0, :, :],cmap=disease_cmap, vmin=0, vmax=3)
    ax.set_title(f'Iteration = {0:03d}')
    plt.colorbar(contour, ax=ax)
    fig.savefig('inital_conditions.png')

    # Propagate the solution
    for k in range(maxiter-1):
        #set change to burn
        sick = np.random.rand(nNorth, nEast)
        fatality = np.random.rand(nNorth, nEast)

        #use current step to set next step:
        people[k+1, :, :] = people[k, :, :]

        # burn from north to south
        getsick = (people[k, :-1, :] == 3) & (people[k, 1:, :] == 2) & \
            (sick[1:, :] <= pspread)
        people[k+1, 1:, :][getsick] = 3
        #add chance to die
        dies = getsick & (fatality[1:, :] <= pfatal)
        people[k+1, 1:, :][dies] = 0

        #Burn from south to north.
        getsick = (people[k, :-1, :] == 2) & (people[k, 1:, :] == 3) & \
            (sick[:-1, :] <= pspread)
        people[k+1, :-1, :][getsick] = 3
        #add chance to die
        dies = getsick & (fatality[:-1, :] <= pfatal)
        people[k+1, :-1, :][dies] = 0       

        #Burn from west to east.
        getsick = (people[k, :, 1:] == 2) & (people[k, :, :-1] == 3) & \
            (sick[:, 1:] <= pspread)
        people[k+1, :, 1:][getsick] = 3
        #add chance to die
        dies = getsick & (fatality[:, 1:] <= pfatal)
        people[k+1, :, 1:][dies] = 0

        #Burn in each cardinal direction from east to west.
        getsick = (people[k, :, :-1] == 2) & (people[k, :, 1:] == 3) & \
            (sick[:, :-1] <= pspread)
        people[k+1, :, :-1][getsick] = 3
        #add chance to die
        dies = getsick & (fatality[:, :-1] <= pfatal)
        people[k+1, :, :-1][dies] = 0
            
        
        #set currently burning to bare
        wassick = people[k, :, :] == 3 # find cells that WERE burning
        people[k+1, wassick] = 1     # ... they are now bare

        # plot initial condition
        fig, ax = plt.subplots(1,1)
        contour = ax.matshow(people[k+1, :, :], cmap=disease_cmap, vmin=0, vmax=3)
        ax.set_title(f'Iteration = {k+1:03d}')
        plt.colorbar(contour, ax=ax)

        fig.savefig(f'fig{k:04d}.png')
        plt.close('all')

        nSick = (people[k+1, :,:] == 3).sum()
        if nSick == 0:
            print(f"disease spread completed in {k+1} steps")
            return k+1
        
# survival rate        
def explore_survivalrate():
    ''' Vary survival rate and see how fast disease ends.'''

    # set probability of burning the next cell from 0 to 1 with a step of 0.05
    survival_prob = np.arange(0, 1.05, .05)
    nsteps = np.zeros(survival_prob.size)

    # for each probability of spread, print number of steps to complete
    for i, psurvive in enumerate(survival_prob):
        # calculate survival rate
        pfatal = 1.0 - psurvive
        # if statement doesn't print: maxiter isnt big enough
        print(f"Disease spread for psurvive = {psurvive}, pfatal = {pfatal}")
        nsteps[i] = disease_spread(nNorth=50, nEast=50, maxiter=400, pfatal = 0.2)

    plt.plot(survival_prob, nsteps)
    plt.savefig('prob_nstep_pspread.png')