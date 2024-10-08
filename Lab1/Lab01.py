#!/usr/bin/env python3

'''
This file performs fire/disease spread simulations for Lab 1. 
To get solution for lab 1: run these commands:

    main function tests:
        spread_sim()
        spread_sim(nEast=5)

    helper functions for problems 2 and 3:
        explore_burnrate()
        vary_burn()
        explore_survivalrate()
        vary_immune()
'''
# import needed packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# set colormaps for plots
forest_cmap = ListedColormap(['tan', 'darkgreen', 'crimson'])
disease_cmap = ListedColormap(['black', 'blue', 'darkgreen', 'crimson'])

def spread_sim(simtype='fire', nNorth=3, nEast=3, maxiter=5, pspread=1.0, pbare=0.0, pstart=0.0, pfatal=0.0):
    '''
    This function performs a fire/disease spread simulation.

    Parameters
    ==========
    simtype : str, defaults to 'fire'
        sets the simulation type ('fire' or 'disease')
    nNorth, nEast : integer, defaults to 3
        Set the north-south (i) and east-west (j) size of grid
        Default is 3 squares in each direction
    maxiter : int, defaults to 4
        Set the maximum number of iterations including initial condition
    pspread : float, defaults to 1
        chance for fire to spread
    pbare : float, defaults to 0
        change for cells to start as a bare patch
    pstart : float, defaults to 0
        chance for cells to start on fire
    pfatal: float, defaults to 0
        chance for sick person to die
    '''

    # Create grid and set initial conditions
    grid = np.zeros([maxiter, nNorth, nEast]) + 2
    forest_percent = np.zeros(maxiter)
    bare_percent = np.zeros(maxiter)
    healthy_percent = np.zeros(maxiter)
    dead_percent = np.zeros(maxiter)
    immune_percent = np.zeros(maxiter)    
    total_cells = nNorth * nEast

    # Add option for center start or random start
    if pstart == 0:
        # Set fire to the center of the grid
        istart, jstart = nNorth // 2, nEast // 2
        grid[0, istart, jstart] = 3

    else:
        # Loop to ensure at least one cell is set on fire
        fire_set = False  # Track whether any cell has been set on fire
        while not fire_set:
            for i in range(nNorth):
                for j in range(nEast):
                    # Set fire to random cells
                    if np.random.rand() < pstart:
                        grid[0, i, j] = 3
                        fire_set = True  # At least one cell is set on fire

            # If no cells were set on fire, loop again
            if not fire_set:
                print("No cells set on fire, retrying...")

    # determine if cells are bare to start
    for i in range(nNorth):
        for j in range(nEast):
            # roll dice to see if it's a bare spot
            if np.random.rand() < pbare:
                grid[0, i, j] = 1  

    # Calculate initial percentages
    if simtype == 'fire':
        forest_cells = (grid[0, :, :] == 2).sum()
        forest_percent[0] = (forest_cells / total_cells) * 100

        bare_cells = (grid[0,:,:] == 1).sum()  # Final step calculation
        bare_percent[0] = (bare_cells / total_cells) * 100
    else:
        healthy_cells = (grid[0, :, :] == 2).sum()
        healthy_percent[0] = (healthy_cells / total_cells) * 100

        dead_cells = (grid[0, :, :] == 0).sum()
        dead_percent[0] = (dead_cells / total_cells) * 100

        immune_cells = (grid[0, :, :] == 1).sum()
        immune_percent[0] = (immune_cells / total_cells) * 100          

    # Plot initial condition
    fig, ax = plt.subplots(1, 1)
    # set different plot specs for fire vs disease
    if simtype =='fire':
        contour = ax.matshow(grid[0, :, :],cmap=forest_cmap, vmin=1, vmax=3)
    else:
        contour = ax.matshow(grid[0, :, :],cmap=disease_cmap, vmin=0, vmax=3)
    ax.set_title(f'Iteration = {0:03d}')
    plt.colorbar(contour, ax=ax)
    fig.savefig('initial_conditions.png')

    # Propagate the solution
    for k in range(maxiter-1):
        #set change to burn
        spread_chance = np.random.rand(nNorth, nEast)
        fatality = np.random.rand(nNorth, nEast) if simtype == 'disease' else None

        #use current step to set next step:
        grid[k+1, :, :] = grid[k, :, :]

        # burn from north to south
        spread = (grid[k, 1:, :] == 2) & (grid[k, :-1, :] == 3) & \
                (spread_chance[1:, :] <= pspread)
        grid[k+1, 1:, :][spread] = 3

        #Burn from south to north.
        spread = (grid[k, :-1, :] == 2) & (grid[k, 1:, :] == 3) & \
                (spread_chance[:-1, :] <= pspread)
        grid[k+1, :-1, :][spread] = 3

        #Burn from west to east.
        spread = (grid[k, :, 1:] == 2) & (grid[k, :, :-1] == 3) & \
                (spread_chance[:, 1:] <= pspread)
        grid[k+1, :, 1:][spread] = 3

        #Burn in each cardinal direction from east to west.
        spread = (grid[k, :, :-1] == 2) & (grid[k, :, 1:] == 3) & \
                (spread_chance[:, :-1] <= pspread)
        grid[k+1, :, :-1][spread] = 3

        # add chance for sick person to die after spreading
        if simtype == 'disease':
            # Add chance for sick person to die
            dies = (grid[k+1, :, :] == 3) & (fatality <= pfatal)
            grid[k+1, dies] = 0

        #set currently burning to bare (also for disease: currently sick to immune)
        wasburn = grid[k, :, :] == 3 # find cells that WERE burning
        grid[k+1, wasburn] = 1     # ... they are now bare

        # plot initial condition
        fig, ax = plt.subplots(1,1)
        if simtype =='fire':
            contour = ax.matshow(grid[k+1, :, :],cmap=forest_cmap, vmin=1, vmax=3)
        else:
            contour = ax.matshow(grid[k+1, :, :],cmap=disease_cmap, vmin=0, vmax=3)
        ax.set_title(f'Iteration = {k+1:03d}')
        plt.colorbar(contour, ax=ax)
        fig.savefig(f'fig{k+1:04d}.png')
        plt.close('all')

        # Calculate the percentage of forested cells for this iteration
        if simtype == 'fire':
            forest_cells = (grid[k+1,:,:] == 2).sum()
            forest_percent[k+1] = (forest_cells / total_cells) * 100

            bare_cells = (grid[k+1,:,:] == 1).sum()  # Final step calculation
            bare_percent[k+1] = (bare_cells / total_cells) * 100

        else:
            healthy_cells = (grid[k+1,:,:] == 2).sum()
            healthy_percent[k+1] = (healthy_cells / total_cells) * 100

            dead_cells = (grid[k+1,:,:] == 0).sum()
            dead_percent[k+1] = (dead_cells / total_cells) * 100

            immune_cells = (grid[k+1,:,:] == 1).sum()
            immune_percent[k+1] = (immune_cells / total_cells) * 100

        
        # Check if spreading is complete
        nBurn = (grid[k+1, :, :] == 3).sum()
        if nBurn == 0:
            print(f"{simtype} completed in {k+1} steps")
            if simtype == 'fire':
                return grid, k+1, forest_percent[:k+1], bare_percent[:k+1]  # Return the grid, steps, and forest percent so far
            else: 
                return grid, k+1, healthy_percent[:k+1], dead_percent[:k+1], immune_percent[:k+1]
    
        # If the max iterations are reached, return the full arrays
    if simtype == 'fire':
        forest_cells = (grid[-1,:,:] == 2).sum()  # Final step calculation
        forest_percent[-1] = (forest_cells / total_cells) * 100

        bare_cells = (grid[-1,:,:] == 1).sum()  # Final step calculation
        bare_percent[-1] = (bare_cells / total_cells) * 100

        return grid, maxiter, forest_percent, bare_percent
    else:
        healthy_cells = (grid[-1,:,:] == 2).sum()
        healthy_percent[-1] = (healthy_cells / total_cells) * 100 

        dead_cells = (grid[-1,:,:] == 0).sum()
        dead_percent[-1] = (dead_cells / total_cells) * 100

        immune_cells = (grid[-1,:,:] == 1).sum()
        immune_percent[-1] = (immune_cells / total_cells) * 100  

        return grid, maxiter, healthy_percent, dead_percent, immune_percent
# define burn rate
def explore_burnrate(nNorth=200, nEast=200, maxiter=500, pstart = 0.0005):
    ''' 
    This function varies burn rate to see how fast fire spreads
    
    Parameters
    ==========
    nNorth, nEast : integer, defaults to 200
        Set the north-south (i) and east-west (j) size of grid
        Default is 200 squares in each direction
    maxiter : int, defaults to 500
        Set the maximum number of iterations including initial condition
    pstart : float, defaults to 0.0005
        Sets the chance for initial cells to start as burning
    '''

    # set probability of burning the next cell from 0 to 1 with a step of 0.05
    prob = np.arange(0, 1.05, .05)
    nsteps = np.zeros(prob.size)
    forest_percent = np.zeros(prob.size)
    bare_percent = np.zeros(prob.size)

    # for each probability of spread, print number of steps to complete
    for i, p in enumerate(prob):
        # if statement doesn't print: maxiter isnt big enough
        print(f"Burning for pspread = {p}")
        grid, n_steps, forest_percent_per_iter, bare_percent_per_iter = \
            spread_sim(nNorth=nNorth, nEast=nEast, pspread=p, maxiter=maxiter, pstart=pstart)
        nsteps[i] = n_steps

        # After the fire, calculate the percentage of forested cells (value 2) remaining
        forest_percent[i] = forest_percent_per_iter[-1]
        bare_percent[i] = bare_percent_per_iter[-1]
    # Plot probability vs. number of steps it took
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Probability for Spread')
    ax1.set_ylabel('Iterations', color=color)
    ax1.plot(prob, nsteps, color=color, label='Iterations')
    ax1.set_ylim(0, nsteps.max())
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis to plot forest percentage
    ax2 = ax1.twinx()  
    color = 'tab:green'
    ax2.set_ylabel('% Cells at the End', color=color)
    ax2.plot(prob, forest_percent, color=color, label='% Forested Cells')
    ax2.plot(prob, bare_percent, color = 'brown', label='% Bare Cells')
    ax2.tick_params(axis='y')

    # Title and save the plot
    plt.legend(loc='upper right')
    plt.title('Probability of Spread vs Iterations and % Cell Types')
    fig.tight_layout()  # To prevent overlap
    plt.savefig('prob_nstep_pspread_forest.png')

# vary bare cells
def vary_bare(nNorth=200, nEast=200, maxiter=500, pstart=0.0005):
    ''' 
    This function varies initial bare spots to see how it impacts fire spread.
    
    Parameters
    ==========
    nNorth, nEast : integer, defaults to 200
        Set the north-south (i) and east-west (j) size of grid
        Default is 200 squares in each direction
    maxiter : int, defaults to 500
        Set the maximum number of iterations including initial condition
    pstart : float, defaults to 0.0005
        Sets the chance for initial cells to start as burning
    '''
    # set probability of cell starting as bare
    prob = np.arange(0, 1.05, .05)
    nsteps = np.zeros(prob.size)
    forest_percent = np.zeros(prob.size)
    bare_percent = np.zeros(prob.size)

    for i, p in enumerate(prob):
        print(f"Burning of pbare = {p}")
        grid, n_steps, forest_percent_per_iter, bare_percent_per_iter = \
            spread_sim(nNorth=nNorth, nEast=nEast, pbare=p, maxiter=maxiter, pstart=pstart)
        nsteps[i] = n_steps

        # After the fire, calculate the percentage of forested cells (value 2) remaining
        forest_percent[i] = forest_percent_per_iter[-1]
        bare_percent[i] = bare_percent_per_iter[-1]
    # Plot probability vs. number of steps it took
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Probability for Bare Cell')
    ax1.set_ylabel('Iterations', color=color)
    ax1.plot(prob, nsteps, color=color, label='Iterations')
    ax1.set_ylim(0, nsteps.max())
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis to plot forest percentage
    ax2 = ax1.twinx()  
    color = 'tab:green'
    ax2.set_ylabel('% Cells at the End', color=color)
    ax2.plot(prob, forest_percent, color=color, label='% Forested Cells')
    ax2.plot(prob, bare_percent, color='brown', label='% Bare Cells')
    ax2.tick_params(axis='y')

    # Title and save the plot
    plt.legend(loc='upper right')
    plt.title('Probability of Bare Cell vs Iterations and % Cell Types')
    fig.tight_layout()  # To prevent overlap
    plt.savefig('prob_nstep_pbare_forest.png')

# survival rate        
def explore_survivalrate(nNorth=200, nEast=200, maxiter=500, pstart=0.0005):
    ''' 
    This function varies survival rate to see how disease spreads.
    
    Parameters
    ==========
    nNorth, nEast : integer, defaults to 200
        Set the north-south (i) and east-west (j) size of grid
        Default is 200 squares in each direction
    maxiter : int, defaults to 500
        Set the maximum number of iterations including initial condition
    pstart : float, defaults to 0.0005
        Sets the chance for initial cells to start as sick
    '''

    # set probability of burning the next cell from 0 to 1 with a step of 0.05
    survival_prob = np.arange(0, 1.05, .05)
    nsteps = np.zeros(survival_prob.size)
    healthy_percent = np.zeros(survival_prob.size)
    dead_percent = np.zeros(survival_prob.size)
    immune_percent = np.zeros(survival_prob.size)

    # for each probability of spread, print number of steps to complete
    for i, psurvive in enumerate(survival_prob):
        # calculate survival rate
        pfatal = 1.0 - psurvive
        # if statement doesn't print: maxiter isnt big enough
        print(f"Disease spread for psurvive = {psurvive}, pfatal = {pfatal}")
        grid, n_steps, healthy_percent_per_iter, dead_percent_per_iter, immune_percent_per_iter  \
            = spread_sim(simtype='disease', nNorth=nNorth, nEast=nEast, maxiter=maxiter, pfatal = pfatal, pstart=pstart)
        nsteps[i] = n_steps

        # After the fire, calculate the percentage of forested cells (value 2) remaining
        healthy_percent[i] = healthy_percent_per_iter[-1]
        dead_percent[i] = dead_percent_per_iter[-1]
        immune_percent[i] = immune_percent_per_iter[-1]

    # Plot probability vs. number of steps it took
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Probability for Survival')
    ax1.set_ylabel('Iterations', color=color)
    ax1.plot(survival_prob, nsteps, color=color, label='Iterations')
    ax1.set_ylim(0, nsteps.max())
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis to plot forest percentage
    ax2 = ax1.twinx()
    ax2.set_ylabel('% Cells at the End')

    ax2.plot(survival_prob, healthy_percent, color='green', label='% Healthy Cells')
    ax2.plot(survival_prob, dead_percent, color='black', label='% Dead Cells')
    ax2.plot(survival_prob, immune_percent, color='orange', label='% Immune cells')
    ax2.set_ylim(-1, 101)
    ax2.tick_params(axis='y')

    # Title and save the plot
    plt.legend(loc='upper right')  
    plt.title('Probability of Survival vs Iterations and % Cell Types')
    fig.tight_layout()  # To prevent overlap
    plt.savefig('prob_nstep_psurvival.png')

    # vary bare cells
def vary_immune(nNorth=200, nEast=200, maxiter=500, pstart=0.0005):
    ''' 
    This function varies disease immunity to see how it impacts disease spread.
    
    Parameters
    ==========
    nNorth, nEast : integer, defaults to 200
        Set the north-south (i) and east-west (j) size of grid
        Default is 200 squares in each direction
    maxiter : int, defaults to 500
        Set the maximum number of iterations including initial condition
    pstart : float, defaults to 0.0005
        Sets the chance for initial cells to start as sick
    '''
    # set probability of cell starting as bare
    prob = np.arange(0, 1.05, .05)
    nsteps = np.zeros(prob.size)
    healthy_percent = np.zeros(prob.size)
    dead_percent = np.zeros(prob.size)
    immune_percent = np.zeros(prob.size)

    for i, pimmune in enumerate(prob):
        print(f"disease spread for pbare = {pimmune}")
        grid, n_steps, healthy_percent_per_iter, dead_percent_per_iter, immune_percent_per_iter \
              = spread_sim(simtype='disease', nNorth=nNorth, nEast=nEast, pbare=pimmune, maxiter=maxiter, pstart=pstart)
        
        # Randomly set cells to be immune with probability `pimmune`
        for r in range(nNorth):  # Loop through all rows
            for c in range(nEast):  # Loop through all columns
                if np.random.rand() < pimmune:
                    grid[0, r, c] = 1  # Set cell to immune in the initial grid
        
        nsteps[i] = n_steps

        # After the fire, calculate the percentage of forested cells (value 2) remaining
        healthy_percent[i] = healthy_percent_per_iter[-1]
        dead_percent[i] = dead_percent_per_iter[-1]
        immune_percent[i] = immune_percent_per_iter[-1]
     
    # Plot probability vs. number of steps it took
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Probability for Initially Immune')
    ax1.set_ylabel('Iterations', color=color)
    ax1.plot(prob, nsteps, color=color, label='Iterations')
    ax1.set_ylim(0, nsteps.max())
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis to plot forest percentage
    ax2 = ax1.twinx()
    ax2.set_ylabel('% Cells at the End')

    ax2.plot(prob, healthy_percent, color='green', label='% Healthy Cells')
    ax2.plot(prob, immune_percent, color='orange', label='% Immune cells')
    ax2.plot(prob, dead_percent, color='black', label='% Dead Cells')

    ax2.set_ylim(-1, 101)
    ax2.tick_params(axis='y')

    # Title and save the plot
    plt.legend(loc='upper right')
    plt.title('Probability of Immune Initial Cell vs Iterations and % Cell Types')
    fig.tight_layout()  # To prevent overlap
    plt.savefig('prob_nstep_pimmune.png')