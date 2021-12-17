import trimesh as tm
import numpy as np
import csv
import random
import sys

def load_mesh(path):
    mesh = tm.load(path)   
    return mesh

def calculate_plane_equation(points):
    p1 = points[0]
    p2 = points[1]
    p3 = points[3]
    # These two vectors are in the plane
    v1 = p3-p1
    v2 = p2-p1
    # the cross product is a vector normal to the plane
    cp = np.cross(v1,v2)
    a, b, c = cp
    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d= np.dot(cp,p3)
    print('The equation is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))
    return (a,b,c,d)
 
def calculate_dist(plane, point):
    a,b,c,d = plane
    x1,y1,z1 = point
    dist = np.abs(a*x1+b*y1+c*z1+d) / np.sqrt(a**2+b**2+c**2)
    return dist

def randomize_points(vertices):
    v1 = vertices[0]
    v2 = vertices[1]
    v3 = vertices[2]

    a,b,c,d = calculate_plane_equation(vertices)

def add_noise_to_pts(sampled_pts, mu=0, sigma=5e-2, nm_of_samples=5000):
    for ii in range(sampled_pts.shape[1]):
        sampled_pts[:,ii] += np.random.normal(mu,sigma,nm_of_samples)
    return sampled_pts

def main():
    # === Parameters === #
    nm_of_samples = 10000
    mu = 0 
    sigma = 7e-2 

    # Load mesh
    mesh = load_mesh('data/raw_meshes/ted.obj')

    # Sameple 
    sampled_pts, indexes = tm.sample.sample_surface(mesh, nm_of_samples, face_weight=None)
    
    noisy_pts = add_noise_to_pts(sampled_pts, mu, sigma, nm_of_samples)

    pts_cloud = tm.points.PointCloud(noisy_pts, colors=None, metadata=None)

    sd = tm.proximity.ProximityQuery(mesh)
    ssd = sd.signed_distance(noisy_pts)
    print (ssd)
    
    # pts_cloud.show()

if __name__ == "__main__":
    main()