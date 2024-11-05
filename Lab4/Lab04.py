#!/usr/bin/env python3
'''
This file performs the heat equation/diffusion simulations for Lab 4.
To get the solution for lab 4, run these commands:

I give some change for the commands in order to get the figures in Q3 could be combined together

example()
permafrost(tmax=5*365*24*3600)
permafrost()
T_gradient_Q3()

'''

import numpy as np
import matplotlib.pyplot as plt


def heatdiff(xmax, tmax, dx, dt, c2=1.0, debug=False,\
              permafrost=False, temp_shift=0):
    '''
    Calculates the heat diffusion matrix of specified dimensions (time and space)

    Parameters:
    -----------
    xmax : float
        maximum depth (meters)
    tmax : float
        maximum time (seconds)
    dx : float
        change in depth (meters)
    dt : float
        change in time (seconds)
    c2 : float, default=1.0
        thermal diffusivity (m^2 / s)
    debug : boolean, default=False
        turns on/off debug option
    permafrost : boolean, default=False
        turns on/off permafrost option, which changes boundary conditions
    temp_shift : float, default=0
        temperature shift due to global warming
        
    Returns:
    --------
    xgrid : numpy array of size xmax*dx
        array of ground depth values from the surface to a specified 
        maximum depth (xmax) with a stepsize of dx

    tgrid : numpy array of total time
        array of the timeseries from 0 to tmax with stepsize dt (in seconds)

    U : matrix of size tgrid x xgrid
        matrix containing heat values (Celsius)

    '''

    # 
    if dt > (dx**2 / (2*c2)):
        raise ValueError('dt is too large!')

    # Start by calculating size of array: MxN
    M = int(np.round(xmax / dx + 1))
    N = int(np.round(tmax / dt + 1))

    xgrid, tgrid = np.arange(0, xmax+dx, dx), np.arange(0, tmax+dt, dt)

    # set debug statements
    if debug:
        print(f'Our grid goes from 0 to {xmax}m and 0 to {tmax}s')
        print(f'Our spatial step is {dx} and time step is {dt}')
        print(f'There are {M} points in space and {N} points in time.')
        print('Here is our spatial grid:')
        print(xgrid)
        print('Here is our time grid:')
        print(tgrid)

    # Initialize our data array:
    U = np.zeros((M, N))

    # Set initial and boundary conditions:
    if permafrost==False:
        U[0, :] = 0
        U[-1, :] = 0
        U[:, 0] = 4*xgrid - 4*xgrid**2
    
    if permafrost==True:
        U[0, :] = temp_kanger(tgrid/(60*60*24), temp_shift) # tgrid in days
        U[-1, :] = 5
        U[:, 0] = 0

    # Set our "r" constant.
    r = c2 * dt / dx**2

    # Solve! Forward differnce ahoy.
    for j in range(N-1):
        U[1:-1, j+1] = (1-2*r) * U[1:-1, j] + \
            r*(U[2:, j] + U[:-2, j])
        
    # Return grid and result:
    return xgrid, tgrid, U

def example(xmax=1, tmax=0.2, dx=0.2, dt=0.02, permafrost=False):
    '''
    Replicates the example in the lab description and produces a table to compare

    Parameters:
    -----------
    xmax : float, default=1
        maximum depth (meters)
    tmax : float, default=0.2
        maximum time (seconds)
    dx : float, default=0.2
        change in depth (meters)
    dt : float, defaut=0.02
        change in time (seconds)
    permafrost : boolean, default=False
    '''
    # calculate heat matrix
    x, time, heat = heatdiff(xmax, tmax, dx, dt, permafrost=permafrost)

    # round values for simplicity 
    rounded_heat = np.round(heat, 7)

    # create table with heat values
    fig = plt.figure(figsize=(10,8))
    row_labels = [f"{i}" for i in range(heat.shape[0])]
    col_labels = [f"{j}" for j in range(heat.shape[1])]
    plt.axis('off')
    table = plt.table(cellText = rounded_heat, loc='center', rowLabels=row_labels,\
                    colLabels=col_labels)
    table.scale(1.2,1.2)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.savefig('example_table.png')

    #print()
    #print(heat)

def temp_kanger(t, temp_shift=0):
    '''
    for an array of times in days, return timeseries of surface 
    temperature for Kangerlussuaq, Greenland

    Parameters:
    -----------
    t : numpy array of total time
        array of the timeseries in days 
    temp_shift : float, default=0
        the temperature shift in the case of global warming conditions
        
    Returns:
    --------
    array of surface temperatures for the given timeseries

    '''
    # monthly avg temps
    t_kanger = np.array([-19.7, -21.0, -17.0, -8.4, 2.3, 8.4,
                         10.7, 8.5, 3.1, -6.0, -12.0, -16.9])
    
    # add the temperature shift to the monthly avg temps
    t_kanger = t_kanger + temp_shift
    
    # calculate and return the sinusoidal temperature function
    t_amp = (t_kanger - t_kanger.mean()).max()
    return t_amp*np.sin(((np.pi/180) * t) - (np.pi/2)) + t_kanger.mean()

def permafrost(xmax=100, tmax=(100*365*24*60*60), dx=0.5, dt=24*3600, c2=2.5e-7,\
                permafrost=True, temp_shift=0):
    '''
    For surface temperatures in Kangerlussuaq, calculates the heat diffusion matrix and plots it.
    Also, plots the temperature profile for summer and winter conditions, and prints the depth of 
    the active and permafrost layers.

    Parameters:
    -----------
    xmax : float
        maximum depth (meters)
    tmax : float
        maximum time (seconds)
    dx : float
        change in depth (meters)
    dt : float
        change in time (seconds)
    c2 : float, default=1.0
        thermal diffusivity (m^2 / s)
    permafrost : boolean, default=False
        turns on/off permafrost option, which changes boundary conditions
    temp_shift : float, default=0
        temperature shift due to global warming
    
    '''
    # calculate heat matrix
    x, time, heat = heatdiff(xmax=xmax, tmax=tmax, dx=dx, dt=dt, c2=c2,\
                            permafrost=permafrost, temp_shift=temp_shift)

    # plot colormap
    fig, axs = plt.subplots(1,2, figsize=(15,6))
    map = axs[0].pcolor((time/ (60*60*24*365)), x, heat, cmap='seismic', vmin=-25, vmax=25)
    axs[0].invert_yaxis()
    plt.colorbar(map, ax=axs[0], label='Temperature ($C$)')
    axs[0].set_xlabel('Time (years)')
    axs[0].set_ylabel('Depth (m)')
    axs[0].set_title('Ground Temperature in Kangerlussuaq, Greenland')

    # get winter/summer profile values
    winter = heat[:, -365:].min(axis=1)
    summer = heat[:, -365:].max(axis=1)

    # plot temp profile
    axs[1].plot(summer, x, color='red', label='summer')
    axs[1].plot(winter, x, color='blue', label='winter')
    axs[1].invert_yaxis()
    axs[1].legend()
    axs[1].set_xlabel('Temperature ($C$)')
    axs[1].set_ylabel('Depth (m)')
    axs[1].set_title('Ground Temperature in Kangerlussuaq, Greenland')

    fig.suptitle(f'Temperature shift of {temp_shift}deg C')
    plt.tight_layout()
    plt.savefig(f'permafrost{temp_shift}.png')
    plt.close('all')

    # calculate active and permafrost layer depth
    active_layer_idx = np.argmax(summer < 0)
    active_layer_depth = x[active_layer_idx]
    permafrost_layer_idx = np.argmax(winter > 0)
    permafrost_layer_depth = x[permafrost_layer_idx]

    print()
    print('Depth of active layer:', active_layer_depth, 'm')
    print('Depth of the permafrost layer:', permafrost_layer_depth, 'm')

'''
An important change is here:

For Q3,
To combining the three temperature gradient figures with different T shift, 
I creat a function call T_gradient_Q3 to achieve that. 
This method may be easy for us to observe the comparison among three different T_shift

'''

def T_gradient_Q3(T_shift = [0.5,1,3],xmax=100, tmax=(100*365*24*60*60), permafrost=True,\
                  dx=0.5, dt=24*3600, c2=2.5e-7):
    
    fig, ax = plt.subplots(1,1, figsize=(8,6))
    for i in T_shift:
        x, time, heat = heatdiff(xmax=xmax, tmax=tmax, dx=dx, dt=dt, c2=c2,\
                            permafrost=permafrost,temp_shift=i)
    
        winter = heat[:, -365:].min(axis=1)
        summer = heat[:, -365:].max(axis=1)

        lw = 2
        labelsize = 15
        years = tmax/(365*24*60*60)
        titlesize = 20

        # plot temp profile
        ax.plot(winter, x, label=f'Winter T shift = {i}',linewidth=lw)
        ax.plot(summer, x,label=f'Summer T shift = {i}',linewidth=lw, linestyle='--')
        plt.legend(loc='lower left') 
        # plt.ylim(0,100)
        # plt.yticks(np.arange(0,101, 10))
        ax.invert_yaxis()
    
    plt.title(f"Ground Temperature: Kangerlussuaq, years={years}",fontsize = titlesize)
    plt.xlabel("Temperature(℃)",fontsize = labelsize)
    plt.ylabel("Depth(m)",fontsize = labelsize)
    plt.xlim(-8,6)
    plt.xticks(np.arange(-8, 8, 2))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'Temperature gradient for Q3.png')
    plt.close('all')  