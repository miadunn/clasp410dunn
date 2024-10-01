#!/usr/bin/env python3
'''
This file performs the competition and predator-prey models for Lab 2.
To get the solution for Lab2, run these commands:

Part 1:
    recreate()
    vary_time()

Part 2:
    vary_ic()
    vary_coeffs()

Part 3:
    vary_ic(model='pp')
    vary_coeffs(model='pp')

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def dNdt_comp(t, N, a=1, b=2, c=1, d=3):
    '''
    This function calculates the Lotka-Volterra competition equations for
    two species. Given normalized populations, `N1` and `N2`, as well as 
    the four coefficients representing population growth and decline,
    calculate the time derivatives dN_1/dt and dN_2/dt and return to the
    caller.
    This function accepts `t`, or time, as an input parameter to be
    compliant with Scipy's ODE solver. However, it is not used in this
    function.

    Parameters
    ----------
    t : float
        The current time (not used here).
    N : two-element list
        The current value of N1 and N2 as a list (e.g., [N1, N2]).
    a, b, c, d : float, defaults=1, 2, 1, 3
        The value of the Lotka-Volterra coefficients.
    
    Returns
    -------
    dN1dt, dN2dt : floats
        The time derivatives of `N1` and `N2`.
    '''
    # Here, N is a two-element list such that N1=N[0] and N2=N[1]
    dN1dt = a*N[0]*(1-N[0]) - b*N[0]*N[1]
    dN2dt = c*N[1]*(1-N[1]) - d*N[1]*N[0]
    return dN1dt, dN2dt

def dNdt_pp(t, N, a=1, b=2, c=1, d=3):
    '''
    This function calculates the Lotka-Volterra predator-prey equations for
    two species. Given normalized populations, `N1` and `N2`, as well as 
    the four coefficients representing population growth and decline,
    calculate the time derivatives dN_1/dt and dN_2/dt and return to the
    caller.
    This function accepts `t`, or time, as an input parameter to be
    compliant with Scipy's ODE solver. However, it is not used in this
    function.

    Parameters
    ----------
    t : float
        The current time (not used here).
    N : two-element list
        The current value of N1 and N2 as a list (e.g., [N1, N2]).
    a, b, c, d : float, defaults=1, 2, 1, 3
        The value of the Lotka-Volterra coefficients.
    
    Returns
    -------
    dN1dt, dN2dt : floats
        The time derivatives of `N1` and `N2`.
    '''
    # Here, N is a two-element list such that N1=N[0] and N2=N[1]
    dN1dt = a*N[0] - b*N[0]*N[1]
    dN2dt = -c*N[1] + d*N[0]*N[1]
    return dN1dt, dN2dt

def euler_solve(func, N1_init=.5, N2_init=.5, dt=.1, t_final=100.0, \
              a=1, b=2, c=1, d=3):
    '''
    This function solves two ODEs using Euler's method

    Parameters
    ----------
    func : function
        A python function that takes `time`, [`N1`, `N2`] as inputs and
        returns the time derivative of N1 and N2.
    N1_init : float, default=0.5
        Initial population capacity for population 1. In the case of predator-prey, this is the prey population.
    N2_init : float, default=0.5
        Initial population capacity for population 2. In the case of predator-prey, this is the predator population.
    dt : float, default=0.1
        Timestep for the iterations
    t_final : float, default=100.0
        Maximum time value  

    Returns 
    -------
    t : Numpy array
        Time elapsed in years
    N : Numpy array
        2D array of normalized population density solutions
    '''

    # Initialize
    t = np.arange(0.0, t_final+dt, dt)
    N = np.zeros((2, t.size))
    N[0, 0] = N1_init
    N[1, 0] = N2_init
    
    # loop through time and calculate derivative
    for i in range(t.size-1):
        dNdt = func(t[i], N[:, i])
        N[:, i+1] = N[:, i] + dt * np.array(dNdt)
    
    # return time and solution arrays
    return t, N

def solve_rk8(func, N1_init=.5, N2_init=.5, dt=10, t_final=100.0, \
              a=1, b=2, c=1, d=3):
    '''
    Solve the Lotka-Volterra competition and predator/prey equations using
    Scipy's ODE class and the adaptive step 8th order solver.
    
    Parameters
    ----------
    func : function
        A python function that takes `time`, [`N1`, `N2`] as inputs and
        returns the time derivative of N1 and N2.
    N1_init, N2_init : float
        Initial conditions for `N1` and `N2`, ranging from (0,1]
    dt : float, default=10
        Largest timestep allowed in years.
    t_final : float, default=100
        Integrate until this value is reached, in years.
    a, b, c, d : float, default=1, 2, 1, 3
        Lotka-Volterra coefficient values
    
    Returns
    -------
    time : Numpy array
        Time elapsed in years.
    N1, N2 : Numpy arrays
        Normalized population density solutions.
    '''
    
    # Configure the initial value problem solver
    result = solve_ivp(func, [0, t_final], [N1_init, N2_init], \
                       args=[a, b, c, d], method='DOP853', max_step=dt)
    
    # Perform the integration
    time, N1, N2 = result.t, result.y[0, :], result.y[1, :]

    # Return values to caller.
    return time, N1, N2


def recreate(dt_comp=1, dt_pp=0.05):
    '''
    This is a function to provide plots for both the competition and 
    predator-prey models. 

    Parameters
    ----------
    dt_comp : float, default = 1.0
        Timestep for the competition model
    dt_pp : float, default = 0.05
        Timestep for the predator-prey model
    '''
    # run both methods for competition model
    t_euler, N_euler = euler_solve(dNdt_comp, N1_init=0.3, N2_init=0.6, dt=dt_comp)
    t_rk8, N1_rk8, N2_rk8 = solve_rk8(dNdt_comp, N1_init=0.3, N2_init=0.6)
    
    # Plot the results
    fig, ax = plt.subplots(1,2)
    ax[0].plot(t_euler, N_euler[0, :], label='Euler N1', color='blue')
    ax[0].plot(t_euler, N_euler[1, :], label='Euler N2', color='red')
    ax[0].plot(t_rk8, N1_rk8, label='RK8 N2', color='blue', linestyle='dotted')
    ax[0].plot(t_rk8, N2_rk8, label='RK8 N2', color='red', linestyle='dotted')
    ax[0].set_xlabel(f'Time (years), dt={dt_comp}')
    ax[0].set_ylabel('Population Carrying Capacity')
    ax[0].set_title('Competiton Model')
    ax[0].legend()

    # run both methods for predator-prey model
    t_euler_pp, N_euler_pp = euler_solve(dNdt_pp, N1_init=0.3, N2_init=0.6, dt=dt_pp)
    t_rk8_pp, N1_rk8_pp, N2_rk8_pp = solve_rk8(dNdt_pp, N1_init=0.3, N2_init=0.6)
    
    # Plot the results
    ax[1].plot(t_euler_pp, N_euler_pp[0, :], label='Euler N1', color='blue')
    ax[1].plot(t_euler_pp, N_euler_pp[1, :], label='Euler N2', color='red')
    ax[1].plot(t_rk8_pp, N1_rk8_pp, label='RK8 N2', color='blue', linestyle='dotted')
    ax[1].plot(t_rk8_pp, N2_rk8_pp, label='RK8 N2', color='red', linestyle='dotted')
    ax[1].set_xlabel(f'Time (years), dt = {dt_pp}')
    ax[1].set_ylabel('Population Carrying Capacity')
    ax[1].set_title('Predator Prey Model')
    ax[1].legend()

    fig.tight_layout()
    plt.savefig('recreation.png')


def vary_time(dt_comp=1, dt_pp=0.05):
    '''
    This is a function to provide plots for both the competition and 
    predator-prey models when varying the timesteps. 

    Parameters
    ----------
    dt_comp : float, default = 1.0
        Timestep for the competition model
    dt_pp : float, default = 0.05
        Timestep for the predator-prey model
    '''

    # dt for rk8 doesn't change, so calculate it for all euler dt changes
    t_rk8, N1_rk8, N2_rk8 = solve_rk8(dNdt_comp, N1_init=0.3, N2_init=0.6)
    t_rk8_pp, N1_rk8_pp, N2_rk8_pp = solve_rk8(dNdt_pp, N1_init=0.3, N2_init=0.6)

    # euler dt = original
    t_euler, N_euler = euler_solve(dNdt_comp, N1_init=0.3, N2_init=0.6, dt=dt_comp)
    t_euler_pp, N_euler_pp = euler_solve(dNdt_pp, N1_init=0.3, N2_init=0.6, dt=dt_pp)

    # euler dt = half
    t_euler_half, N_euler_half = euler_solve(dNdt_comp, N1_init=0.3, N2_init=0.6, dt=dt_comp*0.5)
    t_euler_pp_half, N_euler_pp_half = euler_solve(dNdt_pp, N1_init=0.3, N2_init=0.6, dt=dt_pp*0.5)

    # euler dt = quarter
    t_euler_quart, N_euler_quart = euler_solve(dNdt_comp, N1_init=0.3, N2_init=0.6, dt=dt_comp*0.25)
    t_euler_pp_quart, N_euler_pp_quart = euler_solve(dNdt_pp, N1_init=0.3, N2_init=0.6, dt=dt_pp*0.25)

    # euler dt = tenth
    t_euler_tenth, N_euler_tenth = euler_solve(dNdt_comp, N1_init=0.3, N2_init=0.6, dt=dt_comp*0.1)
    t_euler_pp_tenth, N_euler_pp_tenth = euler_solve(dNdt_pp, N1_init=0.3, N2_init=0.6, dt=dt_pp*0.1)

    # euler dt = tenth
    t_euler_hund, N_euler_hund = euler_solve(dNdt_comp, N1_init=0.3, N2_init=0.6, dt=dt_comp*0.01)
    t_euler_pp_hund, N_euler_pp_hund = euler_solve(dNdt_pp, N1_init=0.3, N2_init=0.6, dt=dt_pp*0.01)
    
    # Plot the results
    fig, ax = plt.subplots(5,2, figsize=(8,8))
    ax[0, 0].plot(t_euler, N_euler[0, :], label='Euler N1', color='blue')
    ax[0, 0].plot(t_euler, N_euler[1, :], label='Euler N2', color='red')
    ax[0, 0].plot(t_rk8, N1_rk8, label='RK8 N1', color='blue', linestyle='dotted')
    ax[0, 0].plot(t_rk8, N2_rk8, label='RK8 N2', color='red', linestyle='dotted')
    ax[0, 0].set_title(f'dt = {dt_comp}')
    
    ax[0, 1].plot(t_euler_pp, N_euler_pp[0, :], label='Euler N1', color='blue')
    ax[0, 1].plot(t_euler_pp, N_euler_pp[1, :], label='Euler N2', color='red')
    ax[0, 1].plot(t_rk8_pp, N1_rk8_pp, label='RK8 N1', color='blue', linestyle='dotted')
    ax[0, 1].plot(t_rk8_pp, N2_rk8_pp, label='RK8 N2', color='red', linestyle='dotted')
    ax[0, 1].set_title(f'dt = {dt_pp}')

    ax[1, 0].plot(t_euler_half, N_euler_half[0, :], label='Euler N1', color='blue')
    ax[1, 0].plot(t_euler_half, N_euler_half[1, :], label='Euler N2', color='red')
    ax[1, 0].plot(t_rk8, N1_rk8, label='RK8 N1', color='blue', linestyle='dotted')
    ax[1, 0].plot(t_rk8, N2_rk8, label='RK8 N2', color='red', linestyle='dotted')
    ax[1, 0].set_title(f'dt = {dt_comp*0.5}')

    ax[1, 1].plot(t_euler_pp_half, N_euler_pp_half[0, :], label='Euler N1', color='blue')
    ax[1, 1].plot(t_euler_pp_half, N_euler_pp_half[1, :], label='Euler N2', color='red')
    ax[1, 1].plot(t_rk8_pp, N1_rk8_pp, label='RK8 N1', color='blue', linestyle='dotted')
    ax[1, 1].plot(t_rk8_pp, N2_rk8_pp, label='RK8 N2', color='red', linestyle='dotted')
    ax[1, 1].set_title(f'dt = {dt_pp*0.5}')

    ax[2, 0].plot(t_euler_quart, N_euler_quart[0, :], label='Euler N1', color='blue')
    ax[2, 0].plot(t_euler_quart, N_euler_quart[1, :], label='Euler N2', color='red')
    ax[2, 0].plot(t_rk8, N1_rk8, label='RK8 N1', color='blue', linestyle='dotted')
    ax[2, 0].plot(t_rk8, N2_rk8, label='RK8 N2', color='red', linestyle='dotted')
    ax[2, 0].set_title(f'dt = {dt_comp*0.25}')

    ax[2, 1].plot(t_euler_pp_quart, N_euler_pp_quart[0, :], label='Euler N1', color='blue')
    ax[2, 1].plot(t_euler_pp_quart, N_euler_pp_quart[1, :], label='Euler N2', color='red')
    ax[2, 1].plot(t_rk8_pp, N1_rk8_pp, label='RK8 N1', color='blue', linestyle='dotted')
    ax[2, 1].plot(t_rk8_pp, N2_rk8_pp, label='RK8 N2', color='red', linestyle='dotted')
    ax[2, 1].set_title(f'dt = {dt_pp*0.25}')

    ax[3, 0].plot(t_euler_tenth, N_euler_tenth[0, :], label='Euler N1', color='blue')
    ax[3, 0].plot(t_euler_tenth, N_euler_tenth[1, :], label='Euler N2', color='red')
    ax[3, 0].plot(t_rk8, N1_rk8, label='RK8 N1', color='blue', linestyle='dotted')
    ax[3, 0].plot(t_rk8, N2_rk8, label='RK8 N2', color='red', linestyle='dotted')
    ax[3, 0].set_title(f'dt = {dt_comp*0.1}')

    ax[3, 1].plot(t_euler_pp_tenth, N_euler_pp_tenth[0, :], label='Euler N1', color='blue')
    ax[3, 1].plot(t_euler_pp_tenth, N_euler_pp_tenth[1, :], label='Euler N2', color='red')
    ax[3, 1].plot(t_rk8_pp, N1_rk8_pp, label='RK8 N1', color='blue', linestyle='dotted')
    ax[3, 1].plot(t_rk8_pp, N2_rk8_pp, label='RK8 N2', color='red', linestyle='dotted')
    ax[3, 1].set_title(f'dt = {round(dt_pp*0.1,3)}')

    ax[4, 0].plot(t_euler_hund, N_euler_hund[0, :], label='Euler N1', color='blue')
    ax[4, 0].plot(t_euler_hund, N_euler_hund[1, :], label='Euler N2', color='red')
    ax[4, 0].plot(t_rk8, N1_rk8, label='RK8 N1', color='blue', linestyle='dotted')
    ax[4, 0].plot(t_rk8, N2_rk8, label='RK8 N2', color='red', linestyle='dotted')
    ax[4, 0].set_title(f'dt = {dt_comp*0.01}')

    ax[4, 1].plot(t_euler_pp_hund, N_euler_pp_hund[0, :], label='Euler N1', color='blue')
    ax[4, 1].plot(t_euler_pp_hund, N_euler_pp_hund[1, :], label='Euler N2', color='red')
    ax[4, 1].plot(t_rk8_pp, N1_rk8_pp, label='RK8 N1', color='blue', linestyle='dotted')
    ax[4, 1].plot(t_rk8_pp, N2_rk8_pp, label='RK8 N2', color='red', linestyle='dotted')
    ax[4, 1].set_title(f'dt = {round(dt_pp*0.01,3)}')

    # add title and legend, then save
    plt.suptitle('Competition and Predator-Prey Models')
    fig.supxlabel('time (years)')
    fig.supylabel('Population Capacity')
    # make room at the top for legend
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=4)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('varying_time.png')


def vary_ic(model='comp', N1_init=0.3, N2_init=0.6, a=1, b=2, c=1, d=3):
    '''
    This is a function to provide plots for both models when varying inition conditions.

    Parameters
    ----------
    model : str, default='comp'
        Sets the model type, 'comp' for competition model or 'pp' for predator-prey model
    N1_init : float, default = 0.3
        Initial population capacity for population 1. In the case of predator-prey, this is the prey population.
    N2_init : float, default = 0.6
        Initial population capacity for population 2. In the case of predator-prey, this is the predator population.
    a, b, c, d : int, defaults = 1, 2, 3, 4
        Coefficients for Lotka-Volterra equations

    '''
    # set if statements for each model type
    if model == 'comp':
        func = dNdt_comp
        dt = 1
        title = 'Competition Model with Varying Initial Conditions'
        fig_name = 'Comp_init_cond.png'
    if model == 'pp':
        func = dNdt_pp
        dt = 0.05
        title = 'Predator-Prey Model with Varying Initial Conditions'
        fig_name = 'PP_init_cond.png'

    # run solvers for N1_init slightly smaller
    teN1_1, NeN1_1 = euler_solve(func, N1_init=N1_init-0.1, N2_init=N2_init, dt=dt, t_final=100)
    trN1_1, NrN1_sm, NrN2_1 = solve_rk8(func, N1_init=N1_init-0.1, N2_init=N2_init, t_final=100)
    
    # run solvers for N1_init slightly bigger
    teN1_2, NeN1_2 = euler_solve(func, N1_init=N1_init+0.1, N2_init=N2_init, dt=dt, t_final=100)
    trN1_2, NrN1_bg, NrN2_2 = solve_rk8(func, N1_init=N1_init+0.1, N2_init=N2_init, t_final=100)
    
        # run solvers for N2_init slightly smaller
    teN2_1, NeN2_1 = euler_solve(func, N1_init=N1_init, N2_init=N2_init-0.1, dt=dt, t_final=100)
    trN2_1, NrN1_1, NrN2_sm = solve_rk8(func, N1_init=N1_init, N2_init=N2_init-0.1, t_final=100)
    
    # run solvers for N2_init slightly bigger
    teN2_2, NeN2_2 = euler_solve(func, N1_init=N1_init, N2_init=N2_init+0.1, dt=dt, t_final=100)
    trN2_2, NrN1_2, NrN2_bg = solve_rk8(func, N1_init=N1_init, N2_init=N2_init+0.1, t_final=100)
    
    # plot changes
    fig, ax = plt.subplots(2,2)
    ax[0, 0].plot(teN1_1, NeN1_1[0, :], label='Euler N1', color='blue')
    ax[0, 0].plot(teN1_1, NeN1_1[1, :], label='Euler N2', color='red')
    ax[0, 0].plot(trN1_1, NrN1_sm, label='RK8 N2', color='blue', linestyle='dotted')
    ax[0, 0].plot(trN1_1, NrN2_1, label='RK8 N2', color='red', linestyle='dotted')
    ax[0, 0].set_title(f'N1 = {round(N1_init - 0.1, 2)}, N2 = {N2_init}')

    ax[0, 1].plot(teN1_2, NeN1_2[0, :], label='Euler N1', color='blue')
    ax[0, 1].plot(teN1_2, NeN1_2[1, :], label='Euler N2', color='red')
    ax[0, 1].plot(trN1_2, NrN1_bg, label='RK8 N2', color='blue', linestyle='dotted')
    ax[0, 1].plot(trN1_2, NrN2_2, label='RK8 N2', color='red', linestyle='dotted')
    ax[0, 1].set_title(f'N1 = {N1_init + 0.1}, N2 = {N2_init}')

    ax[1, 0].plot(teN2_1, NeN2_1[0, :], label='Euler N1', color='blue')
    ax[1, 0].plot(teN2_1, NeN2_1[1, :], label='Euler N2', color='red')
    ax[1, 0].plot(trN2_1, NrN1_1, label='RK8 N2', color='blue', linestyle='dotted')
    ax[1, 0].plot(trN2_1, NrN2_sm, label='RK8 N2', color='red', linestyle='dotted')
    ax[1, 0].set_title(f'N1 = {N1_init}, N2 = {N2_init - 0.1}')

    ax[1, 1].plot(teN2_2, NeN2_2[0, :], label='Euler N1', color='blue')
    ax[1, 1].plot(teN2_2, NeN2_2[1, :], label='Euler N2', color='red')
    ax[1, 1].plot(trN2_2, NrN1_2, label='RK8 N2', color='blue', linestyle='dotted')
    ax[1, 1].plot(trN2_2, NrN2_bg, label='RK8 N2', color='red', linestyle='dotted')
    ax[1, 1].set_title(f'N1 = {N1_init}, N2 = {N2_init + 0.1}')

    plt.suptitle(title)
    fig.supxlabel('Time (years)')
    fig.supylabel('Population Capacity')
    # make room at the top for legend
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=4)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(fig_name)

    # for the predator_prey model, also create a phase diagram
    if model == 'pp':
        fig, ax = plt.subplots(2,2)
        ax[0, 0].plot(NeN1_1[0,:], NeN1_1[1, :], label='Euler', color='green')
        ax[0, 0].plot(NrN1_sm, NrN2_1, label='RK8', color='pink')
        ax[0, 0].set_title(f'N1 = {round(N1_init - 0.1, 2)}, N2 = {N2_init}')

        ax[0, 1].plot(NeN1_2[0,:], NeN1_2[1,:], label='Euler', color='green')
        ax[0, 1].plot(NrN1_bg, NrN2_2, label='RK8', color='pink')
        ax[0, 1].set_title(f'N1 = {N1_init + 0.1}, N2 = {N2_init}')

        ax[1, 0].plot(NeN2_1[0, :], NeN2_1[1, :], label='Euler', color='green')
        ax[1, 0].plot(NrN1_1, NrN2_sm, label='RK8', color='pink')
        ax[1, 0].set_title(f'N1 = {N1_init}, N2 = {N2_init - 0.1}')

        ax[1, 1].plot(NeN2_2[0,:], NeN2_2[1,:], label='Euler', color='green')
        ax[1, 1].plot(NrN1_2, NrN2_bg, label='RK8', color='pink')
        ax[1, 1].set_title(f'N1 = {N1_init}, N2 = {N2_init + 0.1}')

        plt.suptitle('Predator-Prey N1 vs N2 with Varying Initial Conditions')
        fig.supxlabel('N1 (Prey) Population')
        fig.supylabel('N2 (Predator) Population')
        # make room at the top for legend
        handles, labels = ax[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=4)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('vary_ic_phase_diagram.png')
    
    # create plot for equilibrium states
    N1_eq = (c*(a-b)) / ((c*a)-(b*d))
    N2_eq = (a*(c-d)) / ((c*a)-(b*d))
    teN1_eq, NeN1_eq = euler_solve(func, N1_init=N1_eq, N2_init=N2_eq, dt=dt, t_final=100)
    trN1_eq, NrN1_eq, NrN2_eq = solve_rk8(func, N1_init=N1_eq, N2_init=N2_eq, t_final=100)
    if model=='comp':
        fig = plt.figure()
        plt.plot(teN1_eq, NeN1_eq[0, :], label='Euler N1', color='blue')
        plt.plot(teN1_eq, NeN1_eq[1, :], label='Euler N2', color='red')
        plt.plot(trN1_eq, NrN1_eq, label='RK8 N2', color='blue', linestyle='dotted')
        plt.plot(trN1_eq, NrN2_eq, label='RK8 N2', color='red', linestyle='dotted')
        plt.title(f'N1 = {N1_eq}, N2 = {N2_eq}')
        plt.xlabel('Time (years)')
        plt.ylabel('Population Carrying Capacity')
        plt.legend()
        plt.savefig('Comp_equilibrium.png')


def vary_coeffs(model='comp', N1_init=0.3, N2_init=0.6, a=1, b=2, c=1, d=3):
    '''
    This is a function to provide plots for each model when varying coefficients.
    If the model type is predator prey, it will also provide a phase diagram.

        Parameters
    ----------
    model : str, default='comp'
        Sets the model type, 'comp' for competition model or 'pp' for predator-prey model
    N1_init : float, default = 0.3
        Initial population capacity for population 1. In the case of predator-prey, this is the prey population.
    N2_init : float, default = 0.6
        Initial population capacity for population 2. In the case of predator-prey, this is the predator population.
    a, b, c, d : int, defaults = 1, 2, 3, 4
        Coefficients for Lotka-Volterra equations
    '''
    # create if statements for each model
    if model == 'comp':
        func = dNdt_comp
        dt = 1
        title = 'Competition Model with Varying Coefficients'
        fig_name = 'Comp_coeffs.png'
    elif model == 'pp':
        func = dNdt_pp
        dt=0.05
        title = 'Predator-Prey Model with Varying Coefficients'
        fig_name = 'PP_coeffs.png'

    # now vary coefficients
    
    # +/- half of a
    teN1, NeN1 = euler_solve(func, N1_init=N1_init, N2_init=N2_init, dt=dt, t_final=100,\
                                      a=a*0.5, b=b, c=c, d=d)
    trN1, NrN1, NrN2 = solve_rk8(func, N1_init=N1_init, N2_init=N2_init, t_final=100,\
                                      a=a*0.5, b=b, c=c, d=d)
    
    fig, ax = plt.subplots(4,2, figsize=(8,8))
    ax[0, 0].plot(teN1, NeN1[0, :], label='Euler N1', color='blue')
    ax[0, 0].plot(teN1, NeN1[1, :], label='Euler N2', color='red')
    ax[0, 0].plot(trN1, NrN1, label='RK8 N2', color='blue', linestyle='dotted')
    ax[0, 0].plot(trN1, NrN2, label='RK8 N2', color='red', linestyle='dotted')
    ax[0, 0].set_title(f'a = {a * 0.5}')

    teN1, NeN1 = euler_solve(func, N1_init=N1_init, N2_init=N2_init, dt=dt, t_final=100,\
                                      a=a*1.5, b=b, c=c, d=d)
    trN1, NrN1, NrN2 = solve_rk8(func, N1_init=N1_init, N2_init=N2_init, t_final=100,\
                                      a=a*1.5, b=b, c=c, d=d)
    
    ax[0, 1].plot(teN1, NeN1[0, :], label='Euler N1', color='blue')
    ax[0, 1].plot(teN1, NeN1[1, :], label='Euler N2', color='red')
    ax[0, 1].plot(trN1, NrN1, label='RK8 N2', color='blue', linestyle='dotted')
    ax[0, 1].plot(trN1, NrN2, label='RK8 N2', color='red', linestyle='dotted')
    ax[0, 1].set_title(f'a = {a * 1.5}')

    # +/- half of b
    teN1, NeN1 = euler_solve(func, N1_init=N1_init, N2_init=N2_init, dt=dt, t_final=100,\
                                      a=a, b=b*0.5, c=c, d=d)
    trN1, NrN1, NrN2 = solve_rk8(func, N1_init=N1_init, N2_init=N2_init, t_final=100,\
                                      a=a, b=b*0.5, c=c, d=d)
    
    ax[1, 0].plot(teN1, NeN1[0, :], label='Euler N1', color='blue')
    ax[1, 0].plot(teN1, NeN1[1, :], label='Euler N2', color='red')
    ax[1, 0].plot(trN1, NrN1, label='RK8 N2', color='blue', linestyle='dotted')
    ax[1, 0].plot(trN1, NrN2, label='RK8 N2', color='red', linestyle='dotted')
    ax[1, 0].set_title(f'b = {b * 0.5}')

    teN1, NeN1 = euler_solve(func, N1_init=N1_init, N2_init=N2_init, dt=dt, t_final=100,\
                                      a=a, b=b*1.5, c=c, d=d)
    trN1, NrN1, NrN2 = solve_rk8(func, N1_init=N1_init, N2_init=N2_init, t_final=100,\
                                      a=a, b=b*1.5, c=c, d=d)
    
    ax[1, 1].plot(teN1, NeN1[0, :], label='Euler N1', color='blue')
    ax[1, 1].plot(teN1, NeN1[1, :], label='Euler N2', color='red')
    ax[1, 1].plot(trN1, NrN1, label='RK8 N2', color='blue', linestyle='dotted')
    ax[1, 1].plot(trN1, NrN2, label='RK8 N2', color='red', linestyle='dotted')
    ax[1, 1].set_title(f'b = {b * 1.5}')

    # +/- half of c
    teN1, NeN1 = euler_solve(func, N1_init=N1_init, N2_init=N2_init, dt=dt, t_final=100,\
                                      a=a, b=b, c=c*0.5, d=d)
    trN1, NrN1, NrN2 = solve_rk8(func, N1_init=N1_init, N2_init=N2_init, t_final=100,\
                                      a=a, b=b, c=c*0.5, d=d)
    
    ax[2, 0].plot(teN1, NeN1[0, :], label='Euler N1', color='blue')
    ax[2, 0].plot(teN1, NeN1[1, :], label='Euler N2', color='red')
    ax[2, 0].plot(trN1, NrN1, label='RK8 N2', color='blue', linestyle='dotted')
    ax[2, 0].plot(trN1, NrN2, label='RK8 N2', color='red', linestyle='dotted')
    ax[2, 0].set_title(f'c = {c * 0.5}')

    teN1, NeN1 = euler_solve(func, N1_init=N1_init, N2_init=N2_init, dt=dt, t_final=100,\
                                      a=a, b=b, c=c*1.5, d=d)
    trN1, NrN1, NrN2 = solve_rk8(func, N1_init=N1_init, N2_init=N2_init, t_final=100,\
                                      a=a, b=b, c=c*1.5, d=d)
    
    ax[2, 1].plot(teN1, NeN1[0, :], label='Euler N1', color='blue')
    ax[2, 1].plot(teN1, NeN1[1, :], label='Euler N2', color='red')
    ax[2, 1].plot(trN1, NrN1, label='RK8 N2', color='blue', linestyle='dotted')
    ax[2, 1].plot(trN1, NrN2, label='RK8 N2', color='red', linestyle='dotted')
    ax[2, 1].set_title(f'c = {c * 1.5}')

    # +/- half of d
    teN1, NeN1 = euler_solve(func, N1_init=N1_init, N2_init=N2_init, dt=dt, t_final=100,\
                                      a=a, b=b, c=c, d=d*0.5)
    trN1, NrN1, NrN2 = solve_rk8(func, N1_init=N1_init, N2_init=N2_init, t_final=100,\
                                      a=a, b=b, c=c, d=d*0.5)
    
    ax[3, 0].plot(teN1, NeN1[0, :], label='Euler N1', color='blue')
    ax[3, 0].plot(teN1, NeN1[1, :], label='Euler N2', color='red')
    ax[3, 0].plot(trN1, NrN1, label='RK8 N2', color='blue', linestyle='dotted')
    ax[3, 0].plot(trN1, NrN2, label='RK8 N2', color='red', linestyle='dotted')
    ax[3, 0].set_title(f'd = {d * 0.5}')

    teN1, NeN1 = euler_solve(func, N1_init=N1_init, N2_init=N2_init, dt=dt, t_final=100,\
                                      a=a, b=b, c=c, d=d*1.5)
    trN1, NrN1, NrN2 = solve_rk8(func, N1_init=N1_init, N2_init=N2_init, t_final=100,\
                                      a=a, b=b, c=c, d=d*1.5)
    
    ax[3, 1].plot(teN1, NeN1[0, :], label='Euler N1', color='blue')
    ax[3, 1].plot(teN1, NeN1[1, :], label='Euler N2', color='red')
    ax[3, 1].plot(trN1, NrN1, label='RK8 N2', color='blue', linestyle='dotted')
    ax[3, 1].plot(trN1, NrN2, label='RK8 N2', color='red', linestyle='dotted')
    ax[3, 1].set_title(f'd = {d * 1.5}')

    plt.suptitle(title)
    fig.supxlabel('Time (years)')
    fig.supylabel('Population Capacity')
    # make room at the top for legend
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=4)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(fig_name)
    plt.savefig(fig_name)

    # for the predator_prey model, also create a phase diagram
    if model == 'pp':
        fig, ax = plt.subplots(4,2, figsize=(8,8))

        teN1, NeN1 = euler_solve(func, N1_init=N1_init, N2_init=N2_init, dt=dt, t_final=100,\
                                      a=a*0.5, b=b, c=c, d=d)
        trN1, NrN1, NrN2 = solve_rk8(func, N1_init=N1_init, N2_init=N2_init, t_final=100,\
                                      a=a*0.5, b=b, c=c, d=d)

        ax[0, 0].plot(NeN1[0,:], NeN1[1, :], label='Euler', color='red')
        ax[0, 0].plot(NrN1, NrN2, label='RK8', color='blue')
        ax[0, 0].set_title(f'a = {a*0.5}')

        teN1, NeN1 = euler_solve(func, N1_init=N1_init, N2_init=N2_init, dt=dt, t_final=100,\
                                      a=a*1.5, b=b, c=c, d=d)
        trN1, NrN1, NrN2 = solve_rk8(func, N1_init=N1_init, N2_init=N2_init, t_final=100,\
                                      a=a*1.5, b=b, c=c, d=d)

        ax[0, 1].plot(NeN1[0,:], NeN1[1, :], label='Euler', color='red')
        ax[0, 1].plot(NrN1, NrN2, label='RK8', color='blue')
        ax[0, 1].set_title(f'a = {a*1.5}')

        teN1, NeN1 = euler_solve(func, N1_init=N1_init, N2_init=N2_init, dt=dt, t_final=100,\
                                      a=a, b=b*0.5, c=c, d=d)
        trN1, NrN1, NrN2 = solve_rk8(func, N1_init=N1_init, N2_init=N2_init, t_final=100,\
                                      a=a, b=b*0.5, c=c, d=d)

        ax[1, 0].plot(NeN1[0,:], NeN1[1, :], label='Euler', color='red')
        ax[1, 0].plot(NrN1, NrN2, label='RK8', color='blue')
        ax[1, 0].set_title(f'b = {b*0.5}')

        teN1, NeN1 = euler_solve(func, N1_init=N1_init, N2_init=N2_init, dt=dt, t_final=100,\
                                      a=a, b=b*1.5, c=c, d=d)
        trN1, NrN1, NrN2 = solve_rk8(func, N1_init=N1_init, N2_init=N2_init, t_final=100,\
                                      a=a, b=b*1.5, c=c, d=d)

        ax[1, 1].plot(NeN1[0,:], NeN1[1, :], label='Euler', color='red')
        ax[1, 1].plot(NrN1, NrN2, label='RK8', color='blue')
        ax[1, 1].set_title(f'b = {b*1.5}')

        teN1, NeN1 = euler_solve(func, N1_init=N1_init, N2_init=N2_init, dt=dt, t_final=100,\
                                      a=a, b=b, c=c*0.5, d=d)
        trN1, NrN1, NrN2 = solve_rk8(func, N1_init=N1_init, N2_init=N2_init, t_final=100,\
                                      a=a, b=b, c=c*0.5, d=d)

        ax[2, 0].plot(NeN1[0,:], NeN1[1, :], label='Euler', color='red')
        ax[2, 0].plot(NrN1, NrN2, label='RK8', color='blue')
        ax[2, 0].set_title(f'c = {c*0.5}')

        teN1, NeN1 = euler_solve(func, N1_init=N1_init, N2_init=N2_init, dt=dt, t_final=100,\
                                      a=a, b=b, c=c*1.5, d=d)
        trN1, NrN1, NrN2 = solve_rk8(func, N1_init=N1_init, N2_init=N2_init, t_final=100,\
                                      a=a, b=b, c=c*1.5, d=d)

        ax[2, 1].plot(NeN1[0,:], NeN1[1, :], label='Euler', color='red')
        ax[2, 1].plot(NrN1, NrN2, label='RK8', color='blue')
        ax[2, 1].set_title(f'c = {c*1.5}')

        teN1, NeN1 = euler_solve(func, N1_init=N1_init, N2_init=N2_init, dt=dt, t_final=100,\
                                      a=a, b=b, c=c, d=d*0.5)
        trN1, NrN1, NrN2 = solve_rk8(func, N1_init=N1_init, N2_init=N2_init, t_final=100,\
                                      a=a, b=b, c=c, d=d*0.5)

        ax[3, 0].plot(NeN1[0,:], NeN1[1, :], label='Euler', color='red')
        ax[3, 0].plot(NrN1, NrN2, label='RK8', color='blue')
        ax[3, 0].set_title(f'd = {d*0.5}')

        teN1, NeN1 = euler_solve(func, N1_init=N1_init, N2_init=N2_init, dt=dt, t_final=100,\
                                      a=a, b=b, c=c, d=d*1.5)
        trN1, NrN1, NrN2 = solve_rk8(func, N1_init=N1_init, N2_init=N2_init, t_final=100,\
                                      a=a, b=b, c=c, d=d*1.5)

        ax[3, 1].plot(NeN1[0,:], NeN1[1, :], label='Euler', color='red')
        ax[3, 1].plot(NrN1, NrN2, label='RK8', color='blue')
        ax[3, 1].set_title(f'd = {d*1.5}')

        plt.suptitle('Predator-Prey N1 vs N2 with Varying Coefficients')
        fig.supxlabel('N1 (Prey) Population')
        fig.supylabel('N2 (Predator) Population')
        # make room at the top for legend
        handles, labels = ax[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=4)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('vary_coeffs_phase_diagram.png')

def phase_comparisons(N1_init=0.3, N2_init=0.6, a=1, b=2, c=1, d=3):
    '''
    This is a function to provide a phase diagram for initial conditions. 
    The figure will be used to compare to changes in initial conditions/coefficients.

    Parameters
    ----------
    N1_init : float, default = 0.3
        Initial population capacity for population 1. In the case of predator-prey, this is the prey population.
    N2_init : float, default = 0.6
        Initial population capacity for population 2. In the case of predator-prey, this is the predator population.
    a, b, c, d : int, defaults = 1, 2, 3, 4
        Coefficients for Lotka-Volterra equations
    '''
    # run solvers for normal conditions
    teN1, NeN1 = euler_solve(dNdt_pp, N1_init=N1_init, N2_init=N2_init, dt=0.05, t_final=100,\
                                      a=a, b=b, c=c, d=d)
    trN1, NrN1, NrN2 = solve_rk8(dNdt_pp, N1_init=N1_init, N2_init=N2_init, t_final=100,\
                                      a=a, b=b, c=c, d=d)
    
    fig = plt.figure()
    plt.plot(NeN1[0,:], NeN1[1, :], label='Euler', color='red')
    plt.plot(NrN1, NrN2, label='RK8', color='blue')
    plt.suptitle('Predator-Prey Phase Diagram with Original Conditions')
    plt.title(f'N1 = {N1_init}, N2 = {N2_init}, a = {a}, b = {b}, c = {c}, d = {d}')
    plt.xlabel('N1 (Prey) Population')
    plt.ylabel('N2 (Predator) Population')
    plt.legend()
    plt.savefig('phase_diagram_original.png')