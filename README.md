# DeepSDFLerp

Python software for simulating lerp via DeepSDF. 
The implementation is based on the architecture of DeepSDF (presented in https://arxiv.org/abs/1901.05103).

The main idea is to train a simple MLP that given a point (x,y,z) in R^3 - returns the Signed Distance Function (SDF).
We do that by attaching a latent vector to each mesh (i.e. embedding). This latent space actually allows us then to 
interpolate between any two (or more) meshes.

We trained the model on the Thnigi10K dataset of 3D printable models (https://arxiv.org/abs/1605.04797).

Thus this GUI allows to choose predefined or new meshes and interpolate between their latent vectors to get a (hopefully)
smooth transition between both.

# Prerequisites

For convenience, all the dependencies for the project can be ran via pipenv:

1. To install pipenv:
    > pip install pipenv
2. Make sure the current working directory is the main directory of the project
3. Run:
    > pipenv install
    > pipenv shell

# Implementation Details

The architecutre of the DeepSDF (and the embedding layer used when training - a lookup table that converts mesh index to latent vector)
can be found in `model.py`. 
All the code needed for data preperation (sampling and computing SDF) is in `data.py`. We use the "trimesh" framework to traverse the mesh, 
to sample points around it and to compute the SDF.

# Running DeepSDFLerp



Now any `python` command that runs will run in the installed (pipenv) virtual environment.

To train the DeepSDF - run
    >>> python train.py [args]

To run the Lerp GUI - run
    >>> python lerp_gui.py