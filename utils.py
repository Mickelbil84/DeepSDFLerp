import tqdm
import torch
import mcubes
import trimesh
import numpy as np

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
    n = int(2.0 / eps) # number of samples in each axis
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
