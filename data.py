import os
import pickle
import multiprocessing

import tqdm
import torch
import trimesh
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from torch.utils.data import Dataset

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
MAX_FACE_COUNT = 26000
NUM_THREADS = 10

THINGI10K_IN_DIR = 'data/Thingi10K/raw_meshes'
THINGI10K_OUT_DIR = 'data/sampled_sdf/Thingi10K'

class DeepSDFDataset(Dataset):
    """
    Loads for every mesh its sampels and latent vector
    """
    def __init__(self):
        with open(os.path.join(THINGI10K_OUT_DIR, 'latent.pkl'), 'rb') as fp:
            self.latent_dict = pickle.load(fp)
        self.meshes = [m for m in os.listdir(THINGI10K_OUT_DIR) if '.pkl' not in m][:100]
        self.meshes_df = {}
        print("fetching all meshes to RAM. Warning: very expensive")
        def loop(mesh_name):
            return mesh_name, pd.read_excel(os.path.join(THINGI10K_OUT_DIR, mesh_name))
        pairs = Parallel(n_jobs=-1, verbose=10)(delayed(loop)(mesh_name) for mesh_name in self.meshes)
        for mesh_name, df in pairs:
            self.meshes_df[mesh_name] = df


    def __len__(self):
        return len(self.meshes) * NUM_SAMPLES


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        mesh_idx = idx // NUM_SAMPLES
        sample_idx = idx % NUM_SAMPLES

        # Process and load samples
        samples_df = self.meshes_df[self.meshes[mesh_idx]]
        x = samples_df['x'][sample_idx]
        y = samples_df['y'][sample_idx]
        z = samples_df['z'][sample_idx]
        sdf = samples_df['sdf'][sample_idx]
 
        sample = {
            'xyz': torch.from_numpy(np.array([x,y,z])).float(),
            'sdf': torch.from_numpy(np.array([sdf])).float(),
            'latent': torch.from_numpy(self.latent_dict[self.meshes[mesh_idx].split('.')[0] + '.stl']).float()
        }
        
        return sample

def generate_samples_for_mesh(mesh_path, num_samples=NUM_SAMPLES, alpha=ALPHA, 
    boundary_mu=BOUNDARY_MU, boundary_sigma=BOUNDARY_SIGMA, 
    normal_mu=NORMAL_MU, normal_sigma=NORMAL_SIGMA):
    # Load the mesh and normalize it
    mesh = trimesh.load(mesh_path)
    utils.normalize_mesh(mesh)

    if len(mesh.faces) > MAX_FACE_COUNT:
        return None

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
    return df


def process_mesh(in_dir, out_dir, mesh_name):
    mesh_name_pre, _ = mesh_name.split('.')
    if os.path.isfile(os.path.join(out_dir, mesh_name_pre + '.xlsx')):
        return
    try:
        df =  generate_samples_for_mesh(os.path.join(in_dir, mesh_name))
    except:
        return
    
    if df is not None:
        df.to_excel(os.path.join(out_dir, mesh_name_pre + '.xlsx'))


def create_latent_space_dict(in_dir, latent_dim=LATENT_DIM, latent_mu=LATENT_MU, latent_sigma=LATENT_SIGMA):
    meshes = os.listdir(in_dir)
    d = {}
    for mesh_name in meshes:
        z = np.random.normal(latent_mu, latent_sigma, latent_dim)
        d[mesh_name] = z
    return d


if __name__ == "__main__":
    """
    The main function actually prepares the training data, from stl
    to excel with SDF samples + latent vector mapping
    """
    in_dir = 'data/Thingi10K/raw_meshes'
    out_dir = 'data/sampled_sdf/Thingi10K'
    meshes = os.listdir(in_dir)

    for mesh_name in tqdm.tqdm(meshes):
        process_mesh(in_dir, out_dir, mesh_name)

    latent_dict = create_latent_space_dict(in_dir)
    with open(os.path.join(out_dir, 'latent.pkl'), 'wb') as fp:
        pickle.dump(latent_dict, fp)