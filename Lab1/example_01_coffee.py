#!/usr/bin/env python3

'''
A set of tools for solving Newton's law of cooling, i.e.,

$\frac{d T(t)}{dt} = k \left(T_{env} - T(t) \right)$

...where values and units are defined below.

Within this module are functions to return the analytic solution of the heat
equation, numerically solve it, find the time to arrive at a certain
temperature, and visualize the results.

The following table sets the values and their units used throughout this code:

| Symbol | Units  | Value/Meaning                                            |
|--------|--------|----------------------------------------------------------|
|T(t)    | C or K | Surface temperature of body in question                  |
|T_init  | C or K | Initial temperature of body in question                  |
|T_env   | C or K | Temperature of the ambient environment                   |
|k       | 1/s    | Heat transfer coefficient                                |
|t       | s      | Time in seconds                                          |


'''

# Standard imports:
import numpy as np
import matplotlib.pyplot as plt

# Set the plot style
# (see https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html)
plt.style.use('fivethirtyeight')


def solve_temp(t, k=1/300., T_env=20, T_init=90):
    '''
    For a given scalar or array of times, `t`, return the analytic solution
    for Newton's law of cooling:

    $T(t)=T_env + \left( T(t=0) - T_{env} \right) e^{-kt}$

    ...where all values are defined in the docstring for this module.

    Parameters
    ==========
    t : Numpy array
        Array of times, in seconds, for which solution will be provided.


    Other Parameters
    ================
    k : float
        Heat transfer coefficient, defaults to 1/300. s^-1
    T_env : float
        Ambient environment temperature, defaults to 20°C.
    T_init : float
        Initial temperature of cooling object/mass, defaults to °C

    Returns
    =======
    temp : numpy array
        An array of temperatures corresponding to `t`.
    '''

    return T_env + (T_init - T_env) * np.exp(-k*t)


def time_to_temp(T_target, k=1/300., T_env=20, T_init=90):
    '''
    Given an initial temperature, `T_init`, an ambient temperature, `T_env`,
    and a cooling rate, return the time required to reach a target temperature,
    `T_target`.

        Parameters
    ==========
    T_target : scalar or numpy array
        Target temperature in °C.


    Other Parameters
    ================
    k : float
        Heat transfer coefficient, defaults to 1/300. s^-1
    T_env : float
        Ambient environment temperature, defaults to 20°C.
    T_init : float
        Initial temperature of cooling object/mass, defaults to °C

    Returns
    =======
    t : scalar or numpy array
        Time in s to reach the target temperature(s).
    '''

    return (-1/k) * np.log((T_target - T_env)/(T_init - T_env))


def coffee_problem1(T_env=21.0):
    '''
    This function solves the following problem:

    A cup of coffee is 90°C and is too hot to drink. You like cream in
    your coffee and adding it will cool the coffee instantaneously
    by 5°C. The coffee needs to cool until 60°C before it is
    drinkable. When should you add the creamer to get the coffee drinkable
    fastest- right away, or once it is already cooled to 60°C?

    It does so by setting an arbitrary heat transfer constant and
    plotting the cooling curves over 10 minutes for the creamer case and
    non-creamer case. The time-to-60°C is marked for each case.
    The ambient temperature is set to 21°C.

    Finally, we consider the "smart" solution where we let the coffee
    sit until it cools to 65°C and then pour in the cream to get to 60°C.
    '''

    # Create an initial time array:
    t = np.arange(0, 10*60., 0.1)

    # Create temperature curves for our two cases:
    temp_nocrm = solve_temp(t, T_init=90, T_env=T_env)
    temp_cream = solve_temp(t, T_init=90 - 5, T_env=T_env)

    # Get time-to-drinkable:
    tcool_nocrm = time_to_temp(60., T_env=T_env, T_init=90)
    tcool_cream = time_to_temp(60., T_env=T_env, T_init=85)
    tcool_smart = time_to_temp(65., T_env=T_env, T_init=90)

    # Create a figure and axes object, set custom figure size:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot Case 1: No Cream. Plot time-to-drinkable as vertical line that
    # has the same color as the matching time series line.
    # The label is set with an "f-string". More info on those here:
    # https://github.com/spacecataz/python_syntax/blob/main/primer06_strformat.md
    l1 = ax.plot(t, temp_nocrm, label='No Cream')
    ax.axvline(tcool_nocrm, ls='--', c=l1[0].get_color(), lw=1.5,
               label=f"Drinkable in {tcool_nocrm:.0f}s")
    # Set a vertical line for Case 3: Cream at 65°C.
    # Color matches "no cream" case above.
    ax.axvline(tcool_smart, ls=':', c=l1[0].get_color(), lw=1.5,
               label="Cream at 65$^{{\\circ}}C$: \n"
               + f"Drinkable in {tcool_smart:.0f}s")

    # Repeat for Case 2: Cream Added.
    l2 = ax.plot(t, temp_cream, label='Cream Added')
    ax.axvline(tcool_cream, ls='--', c=l2[0].get_color(), lw=1.5,
               label=f"Drinkable in {tcool_cream:.0f}s")

    # Polish things up a bit.
    # Axes labels:
    ax.set_xlabel('Time ($s$)')
    ax.set_ylabel('Coffee Temperature ($^{\\circ}C$)')
    # Title:
    ax.set_title('Adding Cream Makes Java Sippable Faster')
    # Put the legend in a good spot:
    ax.legend(loc='best')
    # Tighten up margins:
    fig.tight_layout()

    return fig
