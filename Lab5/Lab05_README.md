This is the README file for Lab 5 of Climate 410 of Fall 2024.

The Lab04.py file contains all code necessary to complete Lab 5. It contains the following functions:

    gen_grid()
        From the snowball_functions.py file
        Generates a grid of latitudes.
    
    temp_warm()
        From the snowball_functions.py file
        Creates a temperature profile for an array of latitudes.

    insolation()
        From the snowball_functions.py file
        Calculates annual mean insolation for an array of latitudes.

    snowball_earth()
        Simulates a snowball Earth and provides a temperature array 
        as a function of latitude.

    test_snowball()
        Reproduces the example plot from the lab description.

    vary_parameters()
        Runs the snowball_earth function for a range of diffusivity
        and emissivity values.

    test_parameters()
        Plots the diffusivity and emissivity values that best match 
        the warm earth scenario.

    vary_init()
        Varies initial temperature using a dynamic albedo for 
        ice/snow vs water/ground, as well as a flash freeze earth 
        scenario. Plots these scenarios along with the warm earth
        equilibrium for comparisons.

    multiplier()
        Applies an insolation multiplier and plots global average 
        temperatures vs gamma.

To create the figures in my Lab 5 Report, run the following commands:

    Question 1:
        test_snowball()
    Question 2:
        vary_parameters()
        test_parameters()
    Question 3:
        vary_init()
    Question 4:
        multiplier()