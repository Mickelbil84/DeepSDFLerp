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

Now any `python` command that runs will run in the installed (pipenv) virtual environment.

# Implementation Details

The architecutre of the DeepSDF (and the embedding layer used when training - a lookup table that converts mesh index to latent vector)
can be found in `model.py`. We use torch as the deep learning framework in the project.
All the code needed for data preperation (sampling and computing SDF) is in `data.py`. We use the "trimesh" framework to traverse the mesh, 
to sample points around it and to compute the SDF.
Then we trained the model - in `train.py`.
To reconstruct the mesh we use the marching cubes algorithm. More specifically, we use the following implementation: https://github.com/pmneila/PyMCubes 

Once we have the trained weights we can then use the DeepSDF for various applications. In this project we have
the GUI that allows us to interpolate between meshes - found in `lerp_gui.py`, and a test for latent arithmetic logic - 
found in `latent_arithmetic.py`.
The widget for displaying the 3D mesh was taken from: https://github.com/zishun/pyqt-meshviewer

The model was trained on both 100 and 1000 meshes. The result for 100 meshes is more visually appealing (in our opinion), and is the one
chosen by default. In order to use the 1000 mesh version, set the flag `SMALLER_MODEL` in `constants.py` to False.

# Running DeepSDFLerp

To run the application Lerp GUI, run:
    > python lerp_gui.py

# Generating Data and Training

To generate the training data, make sure you have downloaded the Thingi10K dataset into `data/Thingi10K/raw_meshes`.
Link to downloaing the dataset can be found here: https://ten-thousand-models.appspot.com/
The run:
    > python data.py

To train the DeepSDF, run:
    > python train.py
