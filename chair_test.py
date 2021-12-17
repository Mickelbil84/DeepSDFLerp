import tqdm
import torch
import mcubes
import numpy as np

from model import DeepSDF


def chair(model, x, y, z):
        xyz = torch.from_numpy(np.array([x,y,z])).float()
        latent = torch.from_numpy(np.zeros(256)).float()
        with torch.no_grad():
            val = model(xyz, latent)
        return float(val.cpu().numpy()[0])

if __name__ == "__main__":
    model = DeepSDF()
    model.load_state_dict(torch.load('test2.pth'))

    f = lambda x, y, z: chair(model, x, y, z)
    eps = 0.025
    n = int(2.0 / eps)
    grid = np.zeros((n,n,n))
    for i, x in tqdm.tqdm(enumerate(np.arange(-1.0, 1.0, eps)), total=n):
        for j, y in enumerate(np.arange(-1.0, 1.0, eps)):
            for k, z in enumerate(np.arange(-1.0, 1.0, eps)):
                grid[i,j,k] = chair(model, x, y, z)

    vertices, triangles = mcubes.marching_cubes(grid, 0)
    mcubes.export_mesh(vertices, triangles, "test2.dae", "MyChair")