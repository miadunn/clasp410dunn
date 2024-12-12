#!/usr/bin/env python3

'''
This file performs radiative forcing simulations for my final project. 
To get solution for the project: run these commands:

    test_case()
    vary_time()
    vary_parameters()
    compare_gas()
    equivalent_warming()
'''

import numpy as np
import matplotlib.pyplot as plt

# define constants
c_w = 4218 # J/kg/K
rho_w = 1000 # kg/m^3

def calc_T_prime(lambda_R, dz, Q, delta_t=1):
    '''
    Calculate T' with the differential equation from the lab 
    description document.

    Parameters
    ----------
    lambda_R : float
        Climate sensitivity value
    dz : int
        Depth of the ocean mixing layer
    Q : numpy array
        global, annual mean adjusted radiative forcing as a 
        function of time
    delta_t : float, defaults to 1
        Timestep between values

    Returns
    -------
    T_prime : numpy array
        Transient temperature change over time
    
    '''
    
    # Calculate C_E
    C_E = c_w * rho_w * dz  # J/K/m²

    # Calculate tau_R
    tau_R = C_E * lambda_R  # s
    tau_R /= (60 * 60 * 24 * 365)  # Convert from seconds to years

    # Time array derived from Q length and delta_t
    timesteps = len(Q)
    time = np.arange(0, timesteps * delta_t, delta_t)

    # Initialize T_prime array
    T_prime = np.zeros(timesteps)

    # Main loop to calculate T'(t) using midpoint rule
    for t in range(1, timesteps):
        integral_sum = 0

        # Midpoint rule for integral
        for t_prime in range(t):
            midpoint_time = (t_prime + 0.5) * delta_t  # Adjust for timestep size
            Q_t_prime = Q[t_prime]  # Q(t') value
            # Compute the integrand, scaled by delta_t
            integral = ((Q_t_prime / C_E)* 60*60*24*365) \
                * np.exp(midpoint_time / tau_R) * delta_t
            integral_sum += integral  # Sum the contributions

        # Calculate T'(t) using the integral
        T_prime[t] = np.exp(-time[t] / tau_R) * integral_sum

    return T_prime



def test_case():
    '''
    Tests the calc_T_prime function and the conversion 
    from CO2 concentration to transient temperature change.
    Plots temperature change due to CO2 forcing from 2000 to 2100.
    '''

    # set time array
    delta_t = 1
    time = np.arange(2000, 2101, delta_t)
    
    # create CO2 conc. array
    C_2000 = 369.71 #ppm
    C_2023 = 421.08 

    k = np.log(C_2023 / C_2000) / (2023-2000)

    C_CO2 = 369.71 * np.exp(k * (time - time[0]))
    # convert to radiative forcing
    Q_CO2 = 5.35 * np.log(C_CO2/C_CO2[0])

    # calculate temperature change due to CO2
    T_prime = calc_T_prime(1, 300, Q_CO2, delta_t)

    # plot time vs temp
    plt.figure(figsize=(10, 6))
    plt.plot(time, T_prime)
    plt.xlabel('Year')
    plt.ylabel('Temperature Anomaly (K)')
    plt.title('Temperature Change from 2000 to 2100')

    plt.savefig('test_case.png')

    print(f'Temperature change over 100 years is {T_prime[-1] - T_prime[0]}')

def vary_time():
    '''
    Varies the timestep between concentration values to evaluate it's effect
    on the accuracy of the differential equation function.
    Plots each time variation's temperature over time.
    '''

    # create list of timesteps
    timestep_list = [0.5, 1, 5, 10, 20]
    
    plt.figure(figsize=(10, 6))

    # loop through time values to calculate temps
    for delta_t in timestep_list:
        time = np.arange(2000, 2101, delta_t)

        # create CO2 conc. array
        C_2000 = 369.71 #ppm
        C_2023 = 421.08 

        k = np.log(C_2023 / C_2000) / (2023-2000)
        
        C_CO2 = 369.71 * np.exp(k * (time - time[0]))
        Q_CO2 = 5.35 * np.log(C_CO2 / C_CO2[0])

        # calculate T'(t)
        T_prime = calc_T_prime(1, 300, Q_CO2, delta_t)

        # plot
        plt.plot(time, T_prime, label=f'Timestep = {delta_t} years')

    plt.xlabel('Year')
    plt.ylabel('Temperature Anomaly (K)')
    plt.title('Temperature Change from 2000 to 2100')
    plt.legend()
    plt.savefig('vary_time.png')

def vary_parameters():
    '''
    Varies climate sensitivity and mixing layer depth to compare and plot
    their effects on temperature changes.
    '''
    # create list of climate sensitivity values
    lambda_list = [0.5, 1, 1.5]
    
    plt.figure(figsize=(10, 6))

    # loop through lambda values and calculate T'
    for lam in lambda_list:
        delta_t = 0.5
        time = np.arange(2000, 2101, delta_t)

        # exponential CO2 concentration
        C_2000 = 369.71 #ppm
        C_2023 = 421.08 

        k = np.log(C_2023 / C_2000) / (2023-2000)

        C_CO2 = 369.71 * np.exp(k * (time - time[0]))
        Q_CO2 = 5.35 * np.log(C_CO2 / C_CO2[0])

        # calculate T'(t)
        T_prime = calc_T_prime(lam, 300, Q_CO2, delta_t)

        # plot
        plt.plot(time, T_prime, label=rf'$\lambda_R$ = {lam}')

    plt.xlabel('Year')
    plt.ylabel('Temperature Anomaly (K)')
    plt.title('Temperature Change from 2000 to 2100')
    plt.legend()
    plt.savefig('vary_lam.png')


    # create list of dz values
    dz_list = [100, 300, 500]
    
    plt.figure(figsize=(10, 6))

    # loop through list and calculate T'
    for mixed_lyr in dz_list:
        delta_t = 0.5
        time = np.arange(2000, 2101, delta_t)

        # exponential CO2 concentration
        C_2000 = 369.71 #ppm
        C_2023 = 421.08 

        k = np.log(C_2023 / C_2000) / (2023-2000)

        C_CO2 = 369.71 * np.exp(k * (time - time[0]))
        Q_CO2 = 5.35 * np.log(C_CO2 / C_CO2[0])

        # calculate T'(t)
        T_prime = calc_T_prime(1, mixed_lyr, Q_CO2, delta_t)

        # plot
        plt.plot(time, T_prime, label=f'dz = {mixed_lyr} m')

    plt.xlabel('Year')
    plt.ylabel('Temperature Anomaly (K)')
    plt.title('Temperature Change from 2000 to 2100')
    plt.legend()
    plt.savefig('vary_dz.png')

def calc_f(M, N):
    '''
    Calculate f function for IPCC's radiative forcing equations

    Parameters
    ----------
    M : numpy array
        CH4 values in ppb
    N : numpy array
        N2O values in ppb

    Returns
    -------
    f : numpy array
        interaction term for radiative forcing
    '''
    f = 0.47 * np.log(1 + (2.01*10**-5 * (M*N)**0.75) + \
                      (5.31*10**-15 * M * (M*N)**1.52))
    return f

def compare_gas(lam=1, dz=300):
    '''
    Calculate and compare temperature changes for CO2, CH4, and N2O

    Parameters
    ----------
    lam : float, defaults to 1.
        Climate sensitivity value
    dz : float, defaults to 300
        Depth of the mixed layer in meters.
    '''
    # create time array
    delta_t = 0.5
    time = np.arange(2000, 2101, delta_t)
    
    # calc CO2 radiative forcing
    C_2000 = 369.71 #ppm
    C_2023 = 421.08 

    k = np.log(C_2023 / C_2000) / (2023-2000)
    C_CO2 = C_2000 * np.exp(k * (time - time[0]))
    Q_CO2 = 5.35 * np.log(C_CO2/C_CO2[0])

    # define conc. for CH4
    C_2000 = 1773.22 #ppb
    C_2023 = 1921.76 

    k_CH4 = np.log(C_2023 / C_2000) / (2023-2000)
    C_CH4 = C_2000 * np.exp(k_CH4 * (time - time[0]))

    # define conc. for N2O
    C_2000 = 316.36 #ppb
    C_2023 = 336.69 

    k_N2O = np.log(C_2023 / C_2000) / (2023-2000)
    C_N2O = C_2000 * np.exp(k_N2O * (time - time[0]))

    # calc CH4 forcing
    Q_CH4 = 0.036 * (np.sqrt(C_CH4) - np.sqrt(C_CH4[0])) - \
        (calc_f(C_CH4, C_N2O[0]) - calc_f(C_CH4[0], C_N2O[0]))

    # calc N2O forcing
    Q_N2O = 0.12 * (np.sqrt(C_N2O) - np.sqrt(C_N2O[0])) - \
        (calc_f(C_CH4[0], C_N2O) - calc_f(C_CH4[0], C_N2O[0]))

    # calc temp changes for all
    T_prime_CO2 = calc_T_prime(lam, dz, Q_CO2, delta_t)
    T_prime_CH4 = calc_T_prime(lam, dz, Q_CH4, delta_t)
    T_prime_N2O = calc_T_prime(lam, dz, Q_N2O, delta_t)

    # plot
    plt.figure(figsize=(10,6))
    plt.plot(time, T_prime_CO2, label='CO2')
    plt.plot(time, T_prime_CH4, label='CH4')
    plt.plot(time, T_prime_N2O, label='N2O')
    
    plt.xlabel('Year')
    plt.ylabel('Temperature Anomaly (K)')
    plt.title('Temperature Change from 2000 to 2100')
    plt.legend()
    plt.savefig('GHGs.png')


def equivalent_warming(lam=1, dz=300):
    '''
    Finds the amount of CH4 and N2O required to warm the planet the same 
    amount as CO2, and prints those concentrations.

    Parameters
    ----------
    lam : float, defaults to 1.
        Climate sensitivity value
    dz : float, defaults to 300
        Depth of the mixed layer in meters.
    '''
    # create time array
    delta_t = 0.5
    time = np.arange(2000, 2101, delta_t)
    
    # calc CO2 radiative forcing
    C_2000 = 369.71 #ppm
    C_2023 = 421.08 

    k = np.log(C_2023 / C_2000) / (2023-2000)
    C_CO2 = C_2000 * np.exp(k * (time - time[0]))
    Q_CO2 = 5.35 * np.log(C_CO2/C_CO2[0])

    # define conc. for CH4
    C_2000 = 1773.22 #ppb
    C_2023 = 1921.76 

    k_CH4 = np.log(C_2023 / C_2000) / (2023-2000)
    C_CH4 = C_2000 * np.exp(k_CH4 * (time - time[0]))

    # define conc. for N2O
    C_2000 = 316.36 #ppb
    C_2023 = 336.69 

    k_N2O = np.log(C_2023 / C_2000) / (2023-2000)
    C_N2O = C_2000 * np.exp(k_N2O * (time - time[0]))

    # Calculate temperature anomalies fo CO2
    T_prime_CO2 = calc_T_prime(lam, dz, Q_CO2, delta_t)

    # equivalence scaling
    scale_CH4 = 1 # How much CH4 needed for CO2 equivalent forcing
    scale_N2O = 1 # How much N2O needed for CO2 equivalent forcing

    # initialize convergence criteria
    tolerance = 0.05
    max_iterations = 1000
    iteration = 0

    # loop through concentrations until close to CO2 warming
    while iteration < max_iterations:
        # Scale methane and nitrous oxide concentrations
        C_CH4_equiv = scale_CH4 * C_CH4
        C_N2O_equiv = scale_N2O * C_N2O

        # Calculate equivalent forcings
        Q_CH4_equiv = 0.036 * (np.sqrt(C_CH4_equiv) - np.sqrt(C_CH4_equiv[0])) - \
            (calc_f(C_CH4_equiv, C_N2O_equiv[0]) - calc_f(C_CH4_equiv[0], C_N2O_equiv[0]))

        Q_N2O_equiv = 0.12 * (np.sqrt(C_N2O_equiv) - np.sqrt(C_N2O_equiv[0])) - \
            (calc_f(C_CH4_equiv[0], C_N2O_equiv) - calc_f(C_CH4_equiv[0], C_N2O_equiv[0]))

        # Calculate temperature anomalies
        T_prime_CH4_equiv = calc_T_prime(1, 300, Q_CH4_equiv, delta_t)
        T_prime_N2O_equiv = calc_T_prime(1, 300, Q_N2O_equiv, delta_t)

        # Check convergence
        diff_CH4 = np.abs(T_prime_CH4_equiv - T_prime_CO2).max()
        diff_N2O = np.abs(T_prime_N2O_equiv - T_prime_CO2).max()

        if diff_CH4 <= tolerance and diff_N2O <= tolerance:
            break

        # Adjust scaling factors
        if diff_CH4 > tolerance:
            scale_CH4 *= 1.1
        if diff_N2O > tolerance:
            scale_N2O *= 1.1

        iteration += 1

        if iteration == max_iterations:
            print("Warning: Maximum iterations reached before convergence.")

    # Plot
    plt.figure(figsize=(12, 8))
    plt.plot(time, T_prime_CO2, label='CO2 Warming')
    plt.plot(time, T_prime_CH4_equiv, '--', label='CH4 Equivalent to CO2')
    plt.plot(time, T_prime_N2O_equiv, '--', label='N2O Equivalent to CO2')
    plt.xlabel('Year')
    plt.ylabel('Temperature Anomaly (K)')
    plt.title('Greenhouse Gas Warming and Equivalence')
    plt.legend()
    plt.savefig('equiv_warming.png')

    # print conc values
    print(f'equivalent 2023 CH4 concentration: {np.round(C_CH4_equiv[46]/1000, 3)} ppm')
    print(f'equivalent 2023 N2O concentration: {np.round(C_N2O_equiv[46]/1000, 3)} ppm')


