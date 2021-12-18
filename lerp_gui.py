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

# Setup CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        self.lerpButton.clicked.connect(self.interpolate_press)
    
    def interpolate_press(self):
        # mesh = om.read_trimesh('data/raw_meshes/ted.obj')
        
        # Load points from sampled file
        model = DeepSDF().to(device)
        model.load_state_dict(torch.load('checkpoints/checkpoint_e42.pth'))
        
        # Load latent_dict and mesh list to test stuff
        with open(os.path.join(THINGI10K_OUT_DIR, 'latent.pkl'), 'rb') as fp:
            latent_dict = pickle.load(fp)
        tmp_latent = {}
        for key in latent_dict:
            new_key = key.split('.')[0]
            tmp_latent[new_key] = latent_dict[key]
        latent_dict = tmp_latent
        meshes = [m for m in os.listdir(THINGI10K_OUT_DIR) if '.pkl' not in m][:1000]

        # Access latent either via index in the mesh list
        # or directly with the mesh name string
        mesh_idx = 531
        mesh_str = '101620'
        latent_from_idx = True

        # If we choose mesh_idx, then for comparsion with ground truth
        # print the actual mesh name
        print(meshes[mesh_idx])

        eps = 0.01

        # Fetch latent vector
        if latent_from_idx:
            latent_raw = latent_dict[meshes[mesh_idx].split('.')[0]]
        else:
            latent_raw = latent_dict[mesh_str]

        # Compute the mesh from SDF and output to screen
        mesh = utils.deepsdf_to_mesh(model, latent_raw, eps, device)
        self.gl.set_mesh(mesh)
        

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = DeepSDFLerpGUI()
    gui.show()
    sys.exit(app.exec())