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
        model = DeepSDF()
        model.load_state_dict(torch.load('chairtest.pth'))

        def chair(x):
            xyz = torch.from_numpy(np.array(x)).float()
            latent = torch.from_numpy(np.zeros(256)).float()
            with torch.no_grad():
                val = model(xyz, latent)
            return float(val.cpu().numpy()[0])

        ply = []
        eps = 0.05
        n = int(2.0 / eps)
        grid = np.zeros((n,n,n))
        for i, x in tqdm.tqdm(enumerate(np.arange(-1.0, 1.0, eps)), total=n):
            for j, y in enumerate(np.arange(-1.0, 1.0, eps)):
                for k, z in enumerate(np.arange(-1.0, 1.0, eps)):
                    grid[i,j,k] = chair([x, y, z])
                    
        vertices, triangles = mcubes.marching_cubes(grid, 0)
        mesh = trimesh.Trimesh(vertices, triangles)
        utils.normalize_mesh(mesh)
        self.gl.set_mesh(mesh)
        

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = DeepSDFLerpGUI()
    gui.show()
    sys.exit(app.exec())