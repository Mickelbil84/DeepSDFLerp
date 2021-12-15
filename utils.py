import trimesh
import numpy as np

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
