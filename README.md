# Navier-Stokes Visualisation
solving the Navier-Stokes equation (momentum equation) using the
[pseudo-spectral method](https://en.wikipedia.org/wiki/Pseudo-spectral_method).

## Dependencies
* [FFTW3](http://fftw.org)
* [VTK](https://vtk.org) (see also the [example page](http://lorensen.github.io/VTKExamples/site/))

## Usage
The Makefile is used for a simple test program (without vtk).  
For the main program, use CMakeLists.txt (which includes VTK).

## TODO
* Control configuration with a seperate file
* Write data to file upon exit (planned or aborted) in order to resume later
