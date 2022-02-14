import os

import tqdm
import torch
import trimesh
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from torch.utils.data import Dataset

from constants import *
import utils


class DeepSDFDataset(Dataset):
    """
    Loads for every mesh its sampels, sdf and latent vector id
    """

    def __init__(self):
        # Read all the actual meshes that we sampled to RAM
        # We limit the number of trained mesh to a given constant
        self.meshes = [m for m in os.listdir(
            THINGI10K_OUT_DIR) if '.pkl' not in m][:NUM_MESHES]
        self.meshes_df = {}
        print("fetching all meshes to RAM. Warning: very expensive")

        def loop(mesh_name):
            return mesh_name, pd.read_excel(os.path.join(THINGI10K_OUT_DIR, mesh_name))
        pairs = Parallel(n_jobs=-1, verbose=10)(delayed(loop)
                                                (mesh_name) for mesh_name in self.meshes)
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
            'xyz': torch.from_numpy(np.array([x, y, z])).float(),
            'sdf': torch.from_numpy(np.array([sdf])).float(),
            'idx': mesh_idx
        }

        return sample


def generate_samples_for_mesh(mesh_path, num_samples=NUM_SAMPLES, alpha=ALPHA,
                              boundary_mu=BOUNDARY_MU, boundary_sigma=BOUNDARY_SIGMA,
                              normal_mu=NORMAL_MU, normal_sigma=NORMAL_SIGMA):
    """
    Given a mesh path (and all the needed parameters) - we sample enough samples in its neighborhood
    and in normal distribution and compute their SDF.
    Return all the samples (and their SDF) to a pandas DataFrame.
    """
    # Load the mesh and normalize it
    mesh = trimesh.load(mesh_path)
    return utils.generate_samples_for_loaded_mesh(mesh, num_samples, alpha, boundary_mu, boundary_sigma, normal_mu, normal_sigma)


def process_mesh(in_dir, out_dir, mesh_name):
    """
    Wrap the logic for processing a mesh, i.e. loading and generating samples
    (if we haven't already)
    """
    mesh_name_pre, _ = mesh_name.split('.')
    if os.path.isfile(os.path.join(out_dir, mesh_name_pre + '.xlsx')):
        return
    try:
        df = generate_samples_for_mesh(os.path.join(in_dir, mesh_name))
    except:
        return

    if df is not None:
        df.to_excel(os.path.join(out_dir, mesh_name_pre + '.xlsx'))


if __name__ == "__main__":
    """
    The main function actually prepares the training data, 
    from stl to excel with SDF samples 
    """
    in_dir = 'data/Thingi10K/raw_meshes'
    out_dir = 'data/sampled_sdf/Thingi10K'
    meshes = os.listdir(in_dir)

    for mesh_name in tqdm.tqdm(meshes):
        process_mesh(in_dir, out_dir, mesh_name)
