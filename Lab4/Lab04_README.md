This is the README file for Lab 4 of Climate 410 of Fall 2024.

The Lab04.py file contains all code necessary to complete Lab 4. It contains the following functions:

    heatdiff()
        Solves for a matrix of heat values for given time and space dimensions.

    example()
        Runs the heatdiff() function and produces a table for comparison with the lab description.

    temp_kanger()
        Creates an array of surface temperature values at Kangerlussuaq, Greenland for a given time array in days. Also implements a temperature shift for global warming scenarios.

    permafrost()
        Calculates and plots the heat matrix for Kangerlussuaq, Greenland, and plots the temperature profile of winter and summer scenarios.

To create the figures and print statements in my Lab 4 Report, run the following commands:

    Part 1:
        example()
    Part 2:
        permafrost(tmax=5*365*24*3600)
        permafrost()
    Part 3:
        permafrost(temp_shift=0.5)
        permafrost(temp_shift=1)
        permafrost(temp_shift=3)