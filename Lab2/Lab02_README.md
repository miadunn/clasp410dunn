This is the README file for Lab 2 of Climate 410 of Fall 2024.

The Lab02.py file contains all code necessary to complete Lab 2.

It contains the following functions:

    dNdt_comp()
        Contains equations for Lotka-Volterra competition for resources.

    dNdt_pp()
        Contains equations for Lotka-Volterra predator-prey.

    euler_solve()
        Solves a set of two inputted ODEs by Euler's method
    
    solve_rk8()
        Solves a set of two inputted ODEs by an 8th-order Runge-Kutta

    recreate()
        Produces a figure t recreate the one shown in the lab description.

    vary_time()
        Produces a figure with subplots of varying timesteps in the Euler solver.

    vary_ic()
        Produces a figure with subplots of varying initial conditions.
            Two types of models can be used
            Default is the competition model, or run vary_ic(model='pp') for the predator-prey model
        In the case of the competition model, it will also produce a figure showing the initial conditions resulting in equilibrium.
        In the case of the predator-prey model, it will also produce a phase diagram figure.

    vary_coeffs()
        Produces a figure with sublots of varying coefficients.
            Two types of models can be used
            Default is the competition model, or run vary_coeffs(model='pp') for the predator-prey model
        In the case of the predator-prey model, it will also produce a phase diagram figure.

    phase_comparisons()
        Produces a phase diagram figure with the same conditions as the recreation figure.

To create the figures in my Lab 2 Report, run the following commands:

    Part 1:
        recreate()
        vary_time()

    Part 2:
        vary_ic()
        vary_coeffs()

    Part 3:
        vary_ic(model='pp')
        vary_coeffs(model='pp')
        phase_comparisons()