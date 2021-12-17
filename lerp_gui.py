import os
import sys
from numpy.lib.twodim_base import tri

import tqdm
import torch
import mcubes
import numpy as np
import pandas as pd
import trimesh
from PyQt5 import QtCore, QtWidgets

from gui.gui import Ui_DeepSDFLerpGUI
from gui.mgl_qt_meshviewer import QGLControllerWidget

import utils
from data import *
from model import DeepSDF

class DeepSDFLerpGUI(QtWidgets.QMainWindow, Ui_DeepSDFLerpGUI):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        self.gl = QGLControllerWidget(self)
        self.glFrameLayout.addWidget(self.gl)

        timer = QtCore.QTimer(self)
        timer.setInterval(20)  # period, in milliseconds
        timer.timeout.connect(self.gl.updateGL)
        timer.start()

        self.lerpButton.clicked.connect(self.show_ted)
    
    def show_ted(self):
        # mesh = om.read_trimesh('data/raw_meshes/ted.obj')
        
        # Load points from sampled file
        model = DeepSDF().cuda()
        model.load_state_dict(torch.load('checkpoints/checkpoint_e9.pth'))
        
        with open(os.path.join(THINGI10K_OUT_DIR, 'latent.pkl'), 'rb') as fp:
            latent_dict = pickle.load(fp)
        tmp_latent = {}
        for key in latent_dict:
            new_key = key.split('.')[0]
            tmp_latent[new_key] = latent_dict[key]
        latent_dict = tmp_latent
        meshes = [m for m in os.listdir(THINGI10K_OUT_DIR) if '.pkl' not in m][:1000]
        mesh_idx = 531

        print(meshes[mesh_idx])

        def chair(x):
            xyz = torch.from_numpy(np.array(x)).float().view(-1, 3).cuda()
            latent = torch.from_numpy(latent_dict[meshes[mesh_idx].split('.')[0]]).float().view(-1, 256).cuda()
            with torch.no_grad():
                val = model(xyz, latent)
            return val.cpu().numpy()

        ply = []
        eps = 0.005
        n = int(2.0 / eps)
        grid = np.zeros((n,n,n))
        latent_raw = latent_dict[meshes[mesh_idx].split('.')[0]]
        latent = np.zeros((n, 256))
        for i in range(n):
            latent[i, :] = latent_raw
        latent = torch.from_numpy(latent).float().cuda()
        for i, x in tqdm.tqdm(enumerate(np.arange(-1.0, 1.0, eps)), total=n):
            for j, y in enumerate(np.arange(-1.0, 1.0, eps)):
                batch = np.zeros((n, 3))
                for k, z in enumerate(np.arange(-1.0, 1.0, eps)):
                    batch[k, 0] = x
                    batch[k, 1] = y
                    batch[k, 2] = z
                
                xyz = torch.from_numpy(batch).float().cuda()

                with torch.no_grad():
                    val = model(xyz, latent)
                grid[i,j,:] = val.cpu().numpy()[:, 0]
        
        print(np.max(grid), np.min(grid))
                    
        vertices, triangles = mcubes.marching_cubes(grid, 0)
        mcubes.export_mesh(vertices, triangles, "misc/test.dae", "test")
        mesh = trimesh.Trimesh(vertices, triangles)
        utils.normalize_mesh(mesh)
        self.gl.set_mesh(mesh)
        

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = DeepSDFLerpGUI()
    gui.show()
    sys.exit(app.exec())