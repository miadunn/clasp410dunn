#!/usr/bin/env python3

'''
Draft code for Lab 5: SNOWBALL EARTH!!!
'''

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

plt.style.use('fivethirtyeight')

# Some constants:
radearth = 6357000.  # Earth radius in meters.
mxdlyr = 50.         # depth of mixed layer (m)
sigma = 5.67e-8      # Steffan-Boltzman constant
C = 4.2e6            # Heat capacity of water
rho = 1020           # Density of sea-water (kg/m^3)


def gen_grid(nbins=18):
    '''
    Generate a grid from 0 to 180 lat (where 0 is south pole, 180 is north)
    where each returned point represents the cell center.

    Parameters
    ----------
    nbins : int, defaults to 18
        Set the number of latitude bins.

    Returns
    -------
    dlat : float
        Grid spacing in degrees.
    lats : Numpy array
        Array of cell center latitudes.
    '''

    dlat = 180 / nbins  # Latitude spacing.
    lats = np.arange(0, 180, dlat) + dlat/2.

    # Alternative way to obtain grid:
    # lats = np.linspace(dlat/2., 180-dlat/2, nbins)

    return dlat, lats


def temp_warm(lats_in):
    '''
    Create a temperature profile for modern day "warm" earth.

    Parameters
    ----------
    lats_in : Numpy array
        Array of latitudes in degrees where temperature is required

    Returns
    -------
    temp : Numpy array
        Temperature in Celcius.
    '''

    # Get base grid:
    dlat, lats = gen_grid()

    # Set initial temperature curve:
    T_warm = np.array([-47, -19, -11, 1, 9, 14, 19, 23, 25, 25,
                       23, 19, 14, 9, 1, -11, -19, -47])
    coeffs = np.polyfit(lats, T_warm, 2)

    # Now, return fitting:
    temp = coeffs[2] + coeffs[1]*lats_in + coeffs[0] * lats_in**2

    return temp



def insolation(S0, lats):
    '''
    Given a solar constant (`S0`), calculate average annual, longitude-averaged
    insolation values as a function of latitude.
    Insolation is returned at position `lats` in units of W/m^2.

    Parameters
    ----------
    S0 : float
        Solar constant (1370 for typical Earth conditions.)
    lats : Numpy array
        Latitudes to output insolation. Following the grid standards set in
        the diffusion program, polar angle is defined from the south pole.
        In other words, 0 is the south pole, 180 the north.

    Returns
    -------
    insolation : numpy array
        Insolation returned over the input latitudes.
    '''

    # Constants:
    max_tilt = 23.5   # tilt of earth in degrees

    # Create an array to hold insolation:
    insolation = np.zeros(lats.size)

    #  Daily rotation of earth reduces solar constant by distributing the sun
    #  energy all along a zonal band
    dlong = 0.01  # Use 1/100 of a degree in summing over latitudes
    angle = np.cos(np.pi/180. * np.arange(0, 360, dlong))
    angle[angle < 0] = 0
    total_solar = S0 * angle.sum()
    S0_avg = total_solar / (360/dlong)

    # Accumulate normalized insolation through a year.
    # Start with the spin axis tilt for every day in 1 year:
    tilt = [max_tilt * np.cos(2.0*np.pi*day/365) for day in range(365)]

    # Apply to each latitude zone:
    for i, lat in enumerate(lats):
        # Get solar zenith; do not let it go past 180. Convert to latitude.
        zen = lat - 90. + tilt
        zen[zen > 90] = 90
        # Use zenith angle to calculate insolation as function of latitude.
        insolation[i] = S0_avg * np.sum(np.cos(np.pi/180. * zen)) / 365.

    # Average over entire year; multiply by S0 amplitude:
    insolation = S0_avg * insolation / 365

    return insolation



def snowball_earth(nbins=18, dt=1., tstop=10000, lam=100., spherecorr=True,
                    debug=False, albedo=0.3, emiss=1, S0=1370, dynamic_albedo=False,
                    albedo_ice=0.6, albedo_gnd=0.3, init_temp=None, gamma=None):
    '''
    Perform snowball earth simulation.

    Parameters
    ----------
    nbins : int, defaults to 18
        Number of latitude bins.
    dt : float, defaults to 1
        Timestep in units of years
    tstop : float, defaults to 10,000
        Stop time in years
    lam : float, defaults to 100
        Diffusion coefficient of ocean in m^2/s
    spherecorr : bool, defaults to True
        Use the spherical coordinate correction term. This should always be
        true except for testing purposes.
    debug : bool, defaults to False
        Turn  on or off debug print statements.
    albedo : float, defaults to 0.3
        Set the Earth's albedo
    emiss : float, defaults to 1
        Set ground emissivity. Change to zero to turn off radiative cooling.
    S0 : float, defaults to 1370
        Set incoming solar forcing constant. Change to zero to turn off insolation.

    Returns
    -------
    lats : Numpy array
        Latitude grid in degrees where 0 is the south pole.
    Temp : Numpy array
        Final temperature as a function of latitude.
    '''
    # Get time step in seconds:
    dt_sec = 365 * 24 * 3600 * dt  # Years to seconds.

    # Generate grid:
    dlat, lats = gen_grid(nbins)

    # Get grid spacing in meters.
    dy = radearth * np.pi * dlat / 180.

    # Create initial condition:
    if gamma is not None:
       insol = gamma * insolation(S0, lats) 
    else:
        insol = insolation(S0, lats)

    # initialize temp arrays
    if init_temp is not None:
        Temp = np.full(nbins, init_temp, dtype=np.float64)
    else:
        Temp = temp_warm(lats)
    if debug:
        print('Initial temp = ', Temp)

    # Get number of timesteps:
    nstep = int(tstop / dt)

    # Debug for problem initialization
    if debug:
        print("DEBUG MODE!")
        print(f"Function called for nbins={nbins}, dt={dt}, tstop={tstop}")
        print(f"This results in nstep={nstep} time step")
        print(f"dlat={dlat} (deg); dy = {dy} (m)")
        print("Resulting Lat Grid:")
        print(lats)

    # Build A matrix:
    if debug:
        print('Building A matrix...')
    A = np.identity(nbins) * -2  # Set diagonal elements to -2
    A[np.arange(nbins-1), np.arange(nbins-1)+1] = 1  # Set off-diag elements
    A[np.arange(nbins-1)+1, np.arange(nbins-1)] = 1  # Set off-diag elements
    # Set boundary conditions:
    A[0, 1], A[-1, -2] = 2, 2

    # Build "B" matrix for applying spherical correction:
    B = np.zeros((nbins, nbins))
    B[np.arange(nbins-1), np.arange(nbins-1)+1] = 1  # Set off-diag elements
    B[np.arange(nbins-1)+1, np.arange(nbins-1)] = -1  # Set off-diag elements
    # Set boundary conditions:
    B[0, :], B[-1, :] = 0, 0

    # Set the surface area of the "side" of each latitude ring at bin center.
    Axz = np.pi * ((radearth+50.0)**2 - radearth**2) * np.sin(np.pi/180.*lats)
    dAxz = np.matmul(B, Axz) / (Axz * 4 * dy**2)

    if debug:
        print('A = ', A)
    # Set units of A derp
    A /= dy**2

    # Get our "L" matrix:
    L = np.identity(nbins) - dt_sec * lam * A
    L_inv = np.linalg.inv(L)

    if debug:
        print('Time integrating...')
    
    # initialize albedo 
    albedo = np.full(nbins, albedo)
    
    for i in range(nstep):
        # Add spherical correction term:
        if spherecorr:
            Temp += dt_sec * lam * dAxz * np.matmul(B, Temp)

        # apply dynamic albedo
        if dynamic_albedo:
            loc_ice = Temp <= -10
            albedo[loc_ice] = albedo_ice
            albedo[~loc_ice] = albedo_gnd

        #Apply insolation and radiative losses
        radiative = (1-albedo) * insol - emiss*sigma*(Temp+273.15)**4
        Temp += dt_sec * radiative / (rho*C*mxdlyr)
        
        Temp = np.matmul(L_inv, Temp)

    return lats, Temp


def test_snowball(tstop=10000):
    '''
    Reproduce example plot in lecture/handout.

    Using our DEFAULT values (grid size, diffusion, etc.) and a warm-Earth
    initial condition, plot:
        - Initial condition
        - Plot simple diffusion only
        - Plot simple diffusion + spherical correction
        - Plot simple diff + sphere corr + insolation
    '''
    nbins = 18

    # Generate very simple grid
    # Generate grid:
    dlat, lats = gen_grid(nbins)

    # Create initial condition:
    initial = temp_warm(lats)

    # Get simple diffusion solution:
    lats, t_diff = snowball_earth(tstop=tstop, spherecorr=False, S0=0, emiss=0)

    # Get diffusion + spherical correction:
    lats, t_sphe = snowball_earth(tstop=tstop, S0=0, emiss=0)

    # get diffusion + spherecorr + radiative terms
    lats, t_rad = snowball_earth(tstop=tstop)

    # Create figure and plot!
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(lats, initial, label='Warm Earth Init. Cond.')
    ax.plot(lats, t_diff, label='Simple Diffusion')
    ax.plot(lats, t_sphe, label='Diffusion + Sphere. Corr.')
    ax.plot(lats, t_rad, label='Diffusion + Sphere. Corr. + Radiative')

    ax.set_xlabel('Latitude (0=South Pole)')
    ax.set_ylabel('Temperature ($^{\circ} C$)')

    ax.legend(loc='best')

    fig.tight_layout()
    plt.savefig('test_snowball.png')


def vary_parameters(tstop=10000):
    '''
    
    '''

    # initialize
    lambdas = np.arange(0, 160, 10)
    T_lambda = []

    # calculate temps for varying lambdas
    for i in lambdas:
        lats, temps = snowball_earth(tstop=tstop, lam=i)
        T_lambda.append(temps)

    # plot temps for each lambda
    fig = plt.figure()
    for i in range(lambdas.size):
        plt.plot(lats, T_lambda[i], label=rf'$\lambda$ = {lambdas[i]}')
    # add warm earth to plot
    warm = temp_warm(lats)
    
    plt.xlabel('Latitude')
    plt.ylabel('Temperature (C)')
    plt.title('Latitude vs Temperature')
    plt.plot(lats, warm, color='black', label='warm')
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
    plt.tight_layout()
    plt.savefig('vary_lambda.png')

    # vary emissivity
    emissivities = np.arange(0, 1.1, 0.1)
    T_emiss = []

    # calculate temps for each emissivity
    fig = plt.figure()
    for i in emissivities:
        lats, temps = snowball_earth(tstop=tstop, emiss=i)
        T_emiss.append(temps)

    # plot temps for each emissivity
    for i in range(emissivities.size):
        plt.plot(lats, T_emiss[i], label=rf'$e$ = {np.round(emissivities[i], 2)}')
    # add warm earth
    warm = temp_warm(lats)
    plt.plot(lats, warm, color='black', label='warm')
    
    plt.xlabel('Latitude')
    plt.ylabel('Temperature (C)')
    plt.title('Latitude vs Temperature')
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
    plt.tight_layout()
    plt.savefig('vary_emiss.png')


def test_parameters(tstop=10000):



    lats, t_rad = snowball_earth(tstop=tstop, emiss=0.7, lam=100)
    t_warm = temp_warm(lats)

    fig = plt.figure()
    plt.plot(lats, t_rad, label=f'emiss=0.7, lambda=100')
    plt.plot(lats, t_warm, label='warm')

    plt.xlabel('Latitude')
    plt.ylabel('Temperature (C)')
    plt.title('Latitude vs Temperature')
    plt.legend()
    plt.tight_layout()
    plt.savefig('found_emiss.png')


    lats, t_rad = snowball_earth(tstop=tstop, emiss=0.7, lam=30)
    t_warm = temp_warm(lats)

    fig = plt.figure()
    plt.plot(lats, t_rad, label=f'emiss=0.7, lambda=30')
    plt.plot(lats, t_warm, label='warm')

    plt.xlabel('Latitude')
    plt.ylabel('Temperature (C)')
    plt.title('Latitude vs Temperature')
    plt.legend()
    plt.tight_layout()
    plt.savefig('found_lambda.png')


def vary_init(tstop=10000, emiss=0.7, lam=30):



    lats, t_hot = snowball_earth(tstop=tstop, dynamic_albedo=True, init_temp=60, emiss=emiss, lam=lam)
    lats, t_cold = snowball_earth(tstop=tstop, dynamic_albedo=True, init_temp=-60, emiss=emiss, lam=lam)
    lats, t_frozen = snowball_earth(tstop=tstop, albedo=0.6, emiss=emiss, lam=lam)
    t_warm = temp_warm(lats)

    fig = plt.figure()
    plt.plot(lats, t_hot, label='hot earth')
    plt.plot(lats, t_cold, label='cold earth')
    plt.plot(lats, t_frozen, label='flash frozen earth')
    plt.plot(lats, t_warm, label='warm earth')
    
    plt.xlabel('Latitude')
    plt.ylabel('Temperature (C)')
    plt.title('Latitude vs Temperature')
    plt.legend()
    plt.tight_layout()
    plt.savefig('vary_init_cond.png')


def multiplier(tstop=10000, emiss=0.7, lam=30):


    gamma_values = np.round(np.arange(0.4, 1.45, 0.05).tolist() + np.arange(1.4, 0.35, -0.05).tolist(), 3)
    
    #print(gamma_values)
    global_means = []
    temps = np.full(18, -60, dtype=np.float64)

    for gamma in gamma_values:
        lats, Temp = snowball_earth(tstop=tstop, init_temp=temps, gamma=gamma, 
                                    dynamic_albedo=True, emiss=emiss, lam=lam)
        global_mean_temp = np.mean(Temp)
        global_means.append(global_mean_temp)
        temps = Temp.copy()

    fig = plt.figure()
    plt.plot(gamma_values, global_means, label='decreasing gamma')

    gamma_values = np.round(np.arange(0.4, 1.45, 0.05).tolist(), 3)
    
    #print(gamma_values)
    global_means = []
    temps = np.full(18, -60, dtype=np.float64)

    for gamma in gamma_values:
        lats, Temp = snowball_earth(tstop=tstop, init_temp=temps, gamma=gamma, 
                                    dynamic_albedo=True, emiss=emiss, lam=lam)
        global_mean_temp = np.mean(Temp)
        global_means.append(global_mean_temp)
        temps = Temp.copy()

    plt.plot(gamma_values, global_means, label='increasing gamma')
    
    plt.xlabel(r'$\gamma$')
    plt.ylabel('global mean temperature (C)')
    plt.title('global mean temperatures vs gamma multiplier')
    plt.legend()
    plt.tight_layout()
    plt.savefig('gamma_multiplier.png')