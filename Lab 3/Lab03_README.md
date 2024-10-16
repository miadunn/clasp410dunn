This is the README file for Lab 3 of CLimate 410 of Fall 2024.

The Lab03.py file contains all code necessary to complete Lab 3.
It contains the following functions:

    n_layer_atm()
        Solves the N-layer atmosphere problem and returns the temperature at each layer.

    temp_emissivity()
        Calculates surface temperatures for a range of emissivities and produces a plot.

    emissivity_layers()
        For a set emissivity, increases the number of atmospheric layers until a goal temperature as been reached.

    nuclear_winter()
        Finds the surface temperature for a nuclear winter scenario.

To create the figures and print statements in my Lab 3 Report, run the following commands:

    Part 3:
        temp_emissivity()
        emissivity_layers()

    Part 4:
        emissivity_layers(epsilon=1, S0=2600, goaltemp=700, albedo=0, venus=True)

    Part 5:
        nuclear_winter()