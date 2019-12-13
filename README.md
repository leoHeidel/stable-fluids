# stable-fluids

this is a school project for the class INF574 at Polytechnique Paris. The goal of this project is to have an implementation of the artivle *Stable Fluids* by **Jos Stam** (1999).

For this project we use python 3 and Tensorflow 2.0. This may not be the most logical choice, but is a good exercise. A better choice would be to use openGL. 

## Dependencies

- tensorflow 2
- matlplotlib

## Strutcure

- simulatorMultiDim.py : where all the high level mathematical opperations are made.
- geometry.py : where the operations which are specific to the geometry and the number of dimensions are made, such as gradient, laplacian ...
- renderer.py : for rendering by ray tracing in 3D

- 3d-renderer-demo.ipynb : Demo of the renderer
- demo-stable-fluid.ipynb : Demo of the project
