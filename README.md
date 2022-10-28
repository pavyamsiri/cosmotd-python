# CosmoTD - Python Version #

cosmotd-python is a python library that can be used to generate videos and plots of scalar field simulations, particularly
pertaining to simulations studying cosmological topological defects. This was developed as part of my physics honours research
project in 2022 at the University of Sydney.

## Performance ##

This python library is very slow and so I have written another program in C++ that utilises the GPU in order to do the field
simulation code. That application is much faster however it may not work on other machines and is missing the ability to plot,
and the data must be exported first before it can be plotted. The repository can be found at this *[link](https://github.com/pavyamsiri/cosmotd)*.
