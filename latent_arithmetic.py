"""
Few tests for math in latent space
(e.g. "king" - "man" + "woman" =? "queen")
"""
import os

import torch

import utils
from constants import *
from model import DeepSDF, LatentEmbedding


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OUT_DIR = './misc/latent_arithmetic'

# Tuples of (a,b,c) where we compute a-b+c
ARITHMETIC = [
    ("LEGO", "BOTTLE_CAP_TRIPOD", "ICE_CREAM")
]

if __name__ == "__main__":
    # Load models
    deepsdf = DeepSDF().to(device)
    deepsdf.load_state_dict(torch.load(
        'checkpoints/checkpoint_e30.pth', map_location=device))

    embedding = LatentEmbedding(NUM_MESHES).to(device)
    embedding.load_state_dict(torch.load(
        'checkpoints/checkpoint_e30_emb.pth', map_location=device))

    # First output all the raw meshes from the latent index
    for mesh_name in MESHES:
        latent = embedding(torch.tensor(
            MESHES[mesh_name]).to(device)).detach().cpu()
        mesh = utils.deepsdf_to_mesh(deepsdf, latent, eps=0.05, device=device,
                                     out_collada_path=os.path.join(OUT_DIR, 'ORG', mesh_name+'.dae'))

    # And similarly, compute arithmetics
    for mesh_name_a, mesh_name_b, mesh_name_c in ARITHMETIC:
        latent_a = embedding(torch.tensor(
            MESHES[mesh_name_a]).to(device)).detach().cpu()
        latent_b = embedding(torch.tensor(
            MESHES[mesh_name_b]).to(device)).detach().cpu()
        latent_c = embedding(torch.tensor(
            MESHES[mesh_name_c]).to(device)).detach().cpu()
        latent = latent_a - latent_b + latent_c
        mesh_name = '_'.join(
            [mesh_name_a[:3], mesh_name_b[:3], mesh_name_c[:3]])
        mesh = utils.deepsdf_to_mesh(deepsdf, latent, eps=0.05, device=device,
                                     out_collada_path=os.path.join(OUT_DIR, 'RES', mesh_name+'.dae'))
