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

def fire_spread(nNorth=3, nEast=3, maxiter=4, pspread=1.0):
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
        change for fire to spread
    '''

    # Create forest and set initial conditions
    forest = np.zeros([maxiter, nNorth, nEast]) + 2

    # Set fire! To the center of the forest.
    istart, jstart = nNorth//2, nEast//2
    forest[0, istart, jstart] = 3

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

        #Burn in each cardinal direction from south to north.
        doburn = (forest[k, :-1, :] == 3) & (forest[k, :-1, :] == 2) & \
            (ignite[1:, :] <= pspread)
        forest[k+1, :-1, :][doburn] = 3

        #for i in range(1, nNorth):
            #for j in range(nEast):
                # is the current patch burning AND adjacent forested?
                #if (forest[k, i, j] == 3) & (forest[k, i-1, j] == 2): 
                    #spread fire to new square:
                    #forest[k+1, i-1, j] = 3                

        #Burn in each cardinal direction from west to east.
        for i in range(nNorth):
            for j in range(nEast-1):
                # check if adjacent cell to the right is burning
                if (forest[k, i, j] == 3) & (forest[k, i, j+1] == 2): 
                    #spread fire to new square:
                    forest[k+1, i, j+1] = 3

        #Burn in each cardinal direction from east to west.
        for i in range(nNorth):
            for j in range(1, nEast):
                # check if adjacent cell to the left is burning
                if (forest[k, i, j] == 3) & (forest[k, i, j-1] == 2): 
                    #spread fire to new square:
                    forest[k+1, i, j-1] = 3
            
        
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