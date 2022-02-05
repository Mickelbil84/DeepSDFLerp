import math

import tqdm
import torch
import mcubes
import trimesh
import numpy as np

import model
import torch.nn as nn
import torch.optim as optim
import random
import pickle
import os
import data
import utils

from constants import *


def deepsdf_to_mesh(deepsdf, latent_z, eps, device, out_collada_path=None):
    """
    Use the deep network of SDF and the desired latent variable to
    generate a triangular mesh. You can control the resolution of the
    generated mesh via epsilon - smaller epsilon is more exact but also
    more time and memory consuming.

    Args:
        deepsdf (torch.nn.Module): neural network of the SDF
        latent_z (np.array[(LATENT_DIM,)]): a vector in latent space
        eps (float): epsilon of resolution for marching cubes
        device (float): pytorch device (CPU or GPU) to run computations
        out_collada_path (str, optional): write the collada file from mcubes
    """
    n = math.ceil(int(2.0 / eps)) # number of samples in each axis
    grid = np.zeros((n, n, n))
    
    # Transform the latent to a batch of duplicated latents
    latent = np.zeros((n, LATENT_DIM))
    for i in range(n):
        latent[i, :] = latent_z
    latent = torch.from_numpy(latent).float().to(device)

    # Traverse the volume and compute the SDF
    for i, x in tqdm.tqdm(enumerate(np.arange(-1.0, 1.0, eps)), total=n):
        for j, y in enumerate(np.arange(-1.0, 1.0, eps)):
            # To make things more quick, combine all tested z-values
            # to a single batch to run on the (preferably) GPU
            # (this speeds up computation considerably)
            batch = np.zeros((n, 3))
            for k, z in enumerate(np.arange(-1.0, 1.0, eps)):
                batch[k, 0] = x
                batch[k, 1] = y
                batch[k, 2] = z
            xyz = torch.from_numpy(batch).float().to(device)
            with torch.no_grad():
                val = deepsdf(xyz, latent)
            grid[i,j,:] = val.cpu().numpy()[:, 0]
    
    # Compute marching cubes on volume grid
    # (and export if necessary)
    vertices, triangles = mcubes.marching_cubes(grid, 0)
    if out_collada_path is not None:
        mcubes.export_mesh(vertices, triangles, out_collada_path)
    
    if len(vertices) == 0 or len(triangles) == 0:
        return None

    # Convert to trimesh mesh
    mesh = trimesh.Trimesh(vertices, triangles)
    normalize_mesh(mesh)
    return mesh


def normalize_mesh(mesh):
    """
    Get a trimesh and normalize it to the unit sphere
    This function works in place
    """
    mesh.vertices -= mesh.center_mass
    mesh.vertices /= np.linalg.norm(np.max(mesh.vertices))


def sample_random_points(num_points, mu=0, sigma=1):
    """
    Sample random points with normal distribution
    """
    return np.random.normal(mu, sigma, (num_points, 3))


def sample_random_points_near_mesh(mesh, num_points, mu=0, sigma=5e-2):
    """
    Get a Trimesh and sample points near its boundary
    """
    sampled_pts, _ = trimesh.sample.sample_surface(mesh, num_points, face_weight=None)
    noisy_pts = add_noise_to_pts(sampled_pts, mu, sigma, num_points)
    return noisy_pts


def points_to_mesh_of_triangles(points, eps=0.05):
    """
    Get an array of points, and generate a Trimesh of triangles centered at those points
    FOR DISPLAY PURPOSE ONLY
    """
    triangle_vertices = []
    triangle_faces = []
    idx = 0
    for i in range(points.shape[0]):
        x, y, z = points[i]
        triangle_vertices.append([x, y, z])
        triangle_vertices.append([x+eps/2, y+eps, z])
        triangle_vertices.append([x+eps, y, z])
        triangle_faces.append([idx, idx+1, idx+2])
        idx += 3
    return trimesh.Trimesh(vertices=triangle_vertices, faces=triangle_faces, process=False)



def add_noise_to_pts(sampled_pts, mu=0, sigma=5e-2, nm_of_samples=5000):
    """
    Get samples points and add random normal noise to each one
    """
    for ii in range(sampled_pts.shape[1]):
        sampled_pts[:,ii] += np.random.normal(mu,sigma,nm_of_samples)
    return sampled_pts

def mesh_to_deepSDF(out_collada_path=None):
    """
    Get a mesh, sample it and regenrate the latent vecotor
    1. Load a mesh
    2. Randomize a latent vector
    3. Sample 10k points around the mesh
    4. Compute the sdf and set to numpy array
    5. Freeze all model's weights execpt the latent vector
    6. Feed forward the latent vector.
    7. Optimize on the sdf distance.
    8. Compute loss and backprop to change the latent vector.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent = np.random.normal(LATENT_MU, LATENT_SIGMA, LATENT_DIM)
    latent = torch.from_numpy(latent).float().to(device)
    # Init the nn and load weights
    deepsdf = model.DeepSDF().to(device)
    deepsdf.load_state_dict(torch.load('resources/trained_models/checkpoint_d0.05_e199_l2.pth', map_location=device))
    # Freeze all weights
    for param in deepsdf.parameters():
        param.requires_grad = False
    # Unfreeze the input
    latent.requires_grad = True
    # Some config:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(deepsdf.parameters())
    # Load mesh:
            # Load latent_dict and mesh list to test stuff
    with open(os.path.join(THINGI10K_OUT_DIR, 'latent.pkl'), 'rb') as fp:
        latent_dict = pickle.load(fp)
    tmp_latent = {}
    for key in latent_dict:
        new_key = key.split('.')[0]
        tmp_latent[new_key] = latent_dict[key]
    latent_dict = tmp_latent
    meshes = [m for m in os.listdir(THINGI10K_OUT_DIR) if '.pkl' not in m][:1000]

    latents_dict_keys = latent_dict.keys()
    mesh_str = random.choice(list(latents_dict_keys))
    latent_raw = latent_dict[mesh_str]
    eps = 0.05
    mesh = utils.deepsdf_to_mesh(model, latent_raw, eps, device, 'misc/test3.dae')
    

def main():
    mesh_to_deepSDF()

if __name__ == "__main__":
    main()


