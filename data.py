import os
import pickle
import multiprocessing

import tqdm
import trimesh
import numpy as np
import pandas as pd

import utils

LATENT_DIM = 256
NUM_SAMPLES = 50000
ALPHA = 0.8 # percent of samples close to mesh boundary
BOUNDARY_MU = 0
BOUNDARY_SIGMA = 5e-2
NORMAL_MU = 0
NORMAL_SIGMA = 0.7
LATENT_MU = 0
LATENT_SIGMA = 0.01

NUM_THREADS = 10

def generate_samples_for_mesh(mesh_path, num_samples=NUM_SAMPLES, alpha=ALPHA, 
    boundary_mu=BOUNDARY_MU, boundary_sigma=BOUNDARY_SIGMA, 
    normal_mu=NORMAL_MU, normal_sigma=NORMAL_SIGMA,
    latent_dim=LATENT_DIM, latent_mu=LATENT_MU, latent_sigma=LATENT_SIGMA):
    # Load the mesh and normalize it
    mesh = trimesh.load(mesh_path)
    utils.normalize_mesh(mesh)

    # Sample points near the mesh and in normal distribution
    num_mesh_samples = int(alpha * num_samples)
    samples_near_mesh = utils.sample_random_points_near_mesh(
        mesh, num_mesh_samples, mu=boundary_mu, sigma=boundary_sigma)
    samples_normal = utils.sample_random_points(
        num_samples - num_mesh_samples, mu=normal_mu, sigma=normal_sigma)
    samples = np.concatenate([samples_near_mesh, samples_normal])

    # Compute SDF
    pq = trimesh.proximity.ProximityQuery(mesh)
    sdf = pq.signed_distance(samples)
    
    # Generate pandas dataframe
    df = pd.DataFrame({
        'x': samples[:, 0],
        'y': samples[:, 1],
        'z': samples[:, 2],
        'sdf': sdf
    })

    # Return the samples dataframe and the random latent vector
    return df, np.random.normal(latent_mu, latent_sigma, latent_dim)


def process_mesh(in_dir, out_dir, mesh_name):
    print(mesh_name)
    mesh_name_pre, _ = mesh_name.split('.')
    df, latent_z = generate_samples_for_mesh(os.path.join(in_dir, mesh_name))
    df.to_excel(os.path.join(out_dir, mesh_name_pre + '.xlsx'))
    # with open(os.path.join(out_dir, mesh_name_pre + '_z.pkl')) as fp:
    #     pickle.dump(latent_z, fp)

def process_mesh_wrap(args):
    in_dir, out_dir, mesh_name = args
    process_mesh(in_dir, out_dir, mesh_name)

if __name__ == "__main__":
    in_dir = 'data/Thingi10K/raw_meshes'
    out_dir = 'data/sampled_sdf/Thingi10K'
    meshes = os.listdir(in_dir)

    for mesh_name in tqdm.tqdm(meshes):
        process_mesh(in_dir, out_dir, mesh_name)
    # with multiprocessing.Pool(NUM_THREADS) as pool:
    #     pool.map(process_mesh_wrap, zip([in_dir] * len(meshes), [out_dir] * len(meshes), meshes))