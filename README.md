# stable-fluids

This is a school project for the class INF574 at Polytechnique Paris. The goal of this project is to have an implementation of the article *Stable Fluids* by **Jos Stam** (1999).

For this project we use Python 3 and Tensorflow 2.0. This may not be the most logical choice, but it is a good exercise. A better choice would be to use openGL. 

## Dependencies

- tensorflow 2
- matlplotlib

## Structure

- simulatorMultiDim.py : where all the high level mathematical operations are made.
- geometry.py : where the operations which are specific to the geometry and the number of dimensions are made, such as gradient, laplacian ...
- renderer.py : for rendering by ray tracing in 3D

- 3d-renderer-demo.ipynb : Demo of the renderer
- demo-stable-fluid.ipynb : Demo of the project
- simulator-demo-colorblend.ipynb : Demo of the color blending example
- simulator-demo-image-deform.ipynb : Demo of the texture deformation example
