#!/usr/bin/env python 3
'''

'''

import numpy as np
import matplotlib.pyplot as plt

# define constants
sigma = 5.67*10**-8

def n_layer_atm(N, epsilon, S0=1350, albedo=0.33, debug=False):
    '''
    solve n layer atm problem and return temp at each layer

    Parameters
    ----------
    N : int
        Set the number of layers
    epsilon : float, default=1.0
        Set the emissivity of the atmospheric layers
    albedo : float, default=0.33
        Set the planetary albedo from 0 to 1
    S0 : float, default=1350
        Set the incoming solar shortwave flux in Watts/m^2
    debug : boolean, default=False
        Turn on debug output

    Returns
    -------
    temps : Numpy array of size N+1
        Array of temperatures from the Earth's surface (element 0) through
        each of the N atmospheric layers, from lowest to highest
    '''

    # create matrices
    A = np.zeros([N+1, N+1])
    b = np.zeros(N+1)
    b[0] = -S0/4 * (1-albedo)

    if debug:
        print(f"Populating N+1 x N+1 matrix (N = {N})")

    # populate A matrix
    for i in range(N+1):
        for j in range(N+1):
            if debug:
                print(f"Calculating point i={i}, j={j}")
            # diagonal elements are always -2, except for surface
            if i == j:
                A[i, j] = -1*(i > 0) - 1
            else:
                # the pattern from class
                m = np.abs(j-i) - 1
                A[i, j] = epsilon * (1-epsilon)**m

    # at the surface, epsilon = -1, breaking our pattern
    # divide by epsilon along surface to get correct results   
    #  b/c epsilon = 1 at sfc         
    A[0, 1:] /= epsilon

    # Verify our A matrix.
    if debug:
        print(A)

    # get inverse of A matrix
    Ainv = np.linalg.inv(A)

    # multiply Ainv by b to get fluxes
    fluxes = np.matmul(Ainv, b)

    # convert fluxes to temperatures
    # fluxes for atmospheric layers
    temps = (fluxes/epsilon/sigma)**0.25
    temps[0] = (fluxes[0]/sigma)**0.25 # flux at ground: epsilon=1

    return temps


def temp_emissivity(N=1, de=0.05):
    '''
    For a specified range of emissivities, calculate the surface temp
    of a 1 layer atmosphere and plot results.

    Parameters
    ----------
    N : int
        Number of atmospheric layers, default=1
    de : float
        Change in emissivity values, default=0.05
    
    Returns
    -------

    
    '''
    # create arrary to hold temps
    temp_array = []
    
    # set range of emissivities
    # start with de bc emissivity of 0 returns nan
    emissivities = np.arange(de, 1 + de, de)
    
    # loop through each emmisivity
    for epsilon in emissivities:
        temps = n_layer_atm(N=N, epsilon=epsilon)

        # save the 1st temp value (sfc) and append it to the array
        temp_array.append(temps[0])

    plt.plot(temp_array, emissivities, color='blue')
    plt.xlabel('Surface Temperature (K)')
    plt.ylabel('Emissivity')
    plt.title('Surface Temperature vs Emissivity')
    plt.savefig('temp_emissivity.png')

    closest_temp = None
    closest_emissivity = None
    closest_diff = float('inf')  # Start with an infinitely large difference

    # find emissivity of the temperature of Earth's surface
    for i, temp in enumerate(temp_array):
        diff = abs(temp - 288)
        if diff < closest_diff:
            closest_diff = diff
            closest_temp = temp
            closest_emissivity = emissivities[i]

    print(f'Closest temperature to 288K: {closest_temp} with emissivity: {closest_emissivity}')

    #return temp_array, emissivities

def emissivity_layers(epsilon=0.255, S0=1350, albedo=0.33,maxlayers=50, tolerance=0.5, debug=False):
    N = 0
    temp_profile = []

    while N <= maxlayers:
        temps = n_layer_atm(N=N, epsilon=epsilon, S0=S0, albedo=albedo, debug=debug)
        surface_temp = temps[0]
        temp_profile.append(temps)

        # Debug output
        if debug:
            print(f'N={N}, Surface Temperature={surface_temp}')

        if abs(surface_temp - 288) <= tolerance:
            print(f'N={N} with surface temp={surface_temp}')
            #print(temp_profile[-1])
            temp_heights = temp_profile[-1]

            N_layers = np.arange(0, N+1, 1)

            plt.plot(temp_heights, N_layers)
            plt.xlabel('Temperature (K)')
            plt.ylabel('Altitude (N layers)')
            plt.title('Temperature vs Altitude')
            plt.savefig('emissivity_layers.png')

            return N, surface_temp, temp_heights

        N += 1


    print(f'Maximum layers reached ({maxlayers}) without finding target temperature.')
    return None, None  # Return None if the maximum number of layers is reached without success