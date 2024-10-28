#1/usr/bin/env python3
'''
Tools and methods for solving our heat equation/diffusion
'''

import numpy as np
import matplotlib.pyplot as plt


def heatdiff(xmax, tmax, dx, dt, c2=1.0, tolerance=1e-3, debug=True, permafrost=False):
    '''
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
    c2 : float
        thermal diffusivity (m^2 / s)
    tolerance : 
    

    Returns:
    --------

    '''

    if dt > (dx**2 / (2*c2)):
        raise ValueError('dt is too large!')

    # Start by calculating size of array: MxN
    M = int(np.round(xmax / dx + 1))
    N = int(np.round(tmax / dt + 1))

    xgrid, tgrid = np.arange(0, xmax+dx, dx), np.arange(0, tmax+dt, dt)

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

    # Set initial conditions:
    U[:, 0] = 4*xgrid - 4*xgrid**2

    # Set boundary conditions:
    if permafrost==False:
        U[0, :] = 0
        U[-1, :] = 0
    
    if permafrost==True:
        U[0, :] = temp_kanger(tgrid*60*60*24)
        U[-1, :] = 5

    # Set our "r" constant.
    r = c2 * dt / dx**2

    steady_state = False

    # Solve! Forward differnce ahoy.
    for j in range(N-1):
        U[1:-1, j+1] = (1-2*r) * U[1:-1, j] + \
            r*(U[2:, j] + U[:-2, j])
        
        # set limit for tolerance so the function doesn't run longer than needed
        max_change = np.max(np.abs(U[:,j+1] - U[:, j]))
        # 
        if max_change < tolerance:
            steady_state = True
            print(f'steady state reached after {j} steps')
            break


    # Return grid and result:
    return xgrid, tgrid, U

def example(xmax=1, tmax=0.2, dx=0.2, dt=0.02, permafrost=False):
    x, time, heat = heatdiff(xmax, tmax, dx, dt, permafrost=permafrost)

    rounded_heat = np.round(heat, 7)
    row_labels = [f"{i}" for i in range(heat.shape[0])]
    col_labels = [f"{j}" for j in range(heat.shape[1])]

    fig = plt.figure(figsize=(10,8))
    plt.axis('off')
    table = plt.table(cellText = rounded_heat, loc='center', rowLabels=row_labels,\
                    colLabels=col_labels)
    table.scale(1.2,1.2)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.show()

    print()
    print(heat)

def temp_kanger(t):
    '''
    for an array of times in days, return timeseries of surface 
    temperature for Kangerlussuaq, Greenland
    '''
    # monthly avg temps
    t_kanger = np.array([-19.7, -21.0, -17.0, -8.4, 2.3, 8.4,
                         10.7, 8.5, 3.1, -6.0, -12.0, -16.9])
    
    t_amp = (t_kanger - t_kanger.mean()).max()
    return t_amp*np.sin(((np.pi/180) * t) - (np.pi/2)) + t_kanger.mean()

def permafrost(xmax=10, tmax=(5*365*24*60*60), dx=1, dt=24*3600, c2=2.5e-7, tolerance=1e-3, permafrost=True):
    
    x, time, heat = heatdiff(xmax=xmax, tmax=tmax, dx=dx, dt=dt, c2=c2, tolerance=tolerance, permafrost=permafrost)
    
    # plot
    fig, axs = plt.subplots(1,1)
    map = axs.pcolor((time/ (60*60*24*365)), x, heat, cmap='seismic', vmin=-25, vmax=25)
    axs.invert_yaxis()
    plt.colorbar(map, ax=axs, label='Temperature ($C$)')
    plt.savefig('permafrost.png')
    plt.close('all')
