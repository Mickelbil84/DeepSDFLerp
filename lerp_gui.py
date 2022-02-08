import os
import sys
from numpy.lib.twodim_base import tri
from torch._C import _set_graph_executor_optimize
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
from model import DeepSDF, LatentEmbedding

import random

# Setup CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

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
        self.label_4.clicked.connect(self.interpulation_press)
        self.saveDemoButton.clicked.connect(self.save_interpolation_dict)
        self.loadDemoButton.clicked.connect(self.load_interpolation_dict)
        self.numStepsSpinbox.valueChanged.connect(self.changed_value)  
        self.alpha = 0.0
        self.curr_step = 0
        self.nm_of_steps = 1
        self.step_size = float(1/self.nm_of_steps)
        self.interpolated_dict = {}

        self.choose_random_meshes = True
        
        
    def interpolate_press(self, interpulated=False):
        # mesh = om.read_trimesh('data/raw_meshes/ted.obj')
        
        # Load points from sampled file
        model = DeepSDF().to(device)
        # model.load_state_dict(torch.load('resources/trained_models/embedding_on_100/checkpoint_e98.pth', map_location=device))
        model.load_state_dict(torch.load('checkpoints/checkpoint_e199.pth', map_location=device))
        embedding = LatentEmbedding(1000).to(device)
        # embedding.load_state_dict(torch.load('resources/trained_models/embedding_on_100/checkpoint_e98_emb.pth', map_location=device))
        embedding.load_state_dict(torch.load('checkpoints/checkpoint_e199_emb.pth', map_location=device))
        
        # Load latent_dict and mesh list to test stuff
        with open(os.path.join(THINGI10K_OUT_DIR, 'latent.pkl'), 'rb') as fp:
            latent_dict = pickle.load(fp)
        tmp_latent = {}
        for key in latent_dict:
            new_key = key.split('.')[0]
            tmp_latent[new_key] = latent_dict[key]
        latent_dict = tmp_latent
        meshes = [m for m in os.listdir(THINGI10K_OUT_DIR) if '.pkl' not in m][:NUM_MESHES]

        # Access latent either via index in the mesh list
        # or directly with the mesh name string
        mesh_idx = 1036655#100
        mesh_idx2 = 1036651#100
        # Choose two random meshes for interpolation
        latents_dict_keys = latent_dict.keys()
        if self.choose_random_meshes:
            mesh_str = random.choice(list(latents_dict_keys)[:99])
            mesh_str2 = random.choice(list(latents_dict_keys)[:99])
        else:
            mesh_str = '101620'
            mesh_str2 = '101864'

        latent_from_idx = False

        # # If we choose mesh_idx, then for comparsion with ground truth
        # # print the actual mesh name
        # print(meshes[mesh_idx])

        eps = 0.05

        # Fetch latent vector
        # if latent_from_idx:
        #     latent_raw = latent_dict[meshes[mesh_idx].split('.')[0]]
        #     latent_raw2 = latent_dict[meshes[mesh_idx2].split('.')[0]]
        # else:
        #     latent_raw = latent_dict[mesh_str]
        #     latent_raw2 = latent_dict[mesh_str2]

        mesh_idx = 0
        mesh_idx2 = 1

        for i, m in enumerate(meshes[:1000]):
            print(i, '-', m.split('.')[0])

        print(meshes[mesh_idx])
        print(meshes[mesh_idx2])

        print(torch.tensor(mesh_idx))
        latent_raw = embedding(torch.tensor(mesh_idx).to(device)).detach().cpu()
        latent_raw2 = embedding(torch.tensor(mesh_idx2).to(device)).detach().cpu()

        # Interpolate two meshes with linear interpolation. 
        # save all to a dictionary
        for ii in range(self.nm_of_steps+1):
            # self.alpha = round(float(self.alpha + self.step_size),1)
            self.alpha = ii / self.nm_of_steps
            latent_raw_to_dict = self.alpha*latent_raw + (1-self.alpha)*latent_raw2
            mesh = utils.deepsdf_to_mesh(model, latent_raw_to_dict, eps, device)# 'misc/test3.dae')
            self.interpolated_dict[ii] = mesh

        # if interpulated:
        #     self.alpha = round(float(self.alpha + self.step_size),1)
        #     if self.alpha > 1:
        #         self.alpha = 0.0
        #     print (self.alpha)

        # latent_raw3 = self.alpha*latent_raw + (1-self.alpha)*latent_raw2

        # Compute the mesh from SDF and output to screen
        # mesh = utils.deepsdf_to_mesh(model, latent_raw, eps, device, 'misc/test3.dae')
        self.gl.set_mesh(self.interpolated_dict[0])

    def interpulation_press(self):
        """
            Temporal usage: FIXME[yoni]: change to bar usage.
            By pressing the interpolate button, this function will load the next interpolation out of the dictionary.
            Current step will be updated. 
        """
        # self.interpolate_press(True)
        if self.curr_step >= len(self.interpolated_dict.keys()):
            self.curr_step = 0
        self.gl.set_mesh(self.interpolated_dict[self.curr_step])
        self.curr_step = (self.curr_step + 1) % (self.nm_of_steps+1)
        _translate = QtCore.QCoreApplication.translate
        self.currentStepLabel.setText(_translate("DeepSDFLerpGUI", "{}{}".format("Current step: ", self.curr_step)))
        
    def changed_value(self):
        """
            Stores the value of number of interpolations once it changed
        """
        self.nm_of_steps = self.numStepsSpinbox.value()    
        self.step_size = float(1/self.nm_of_steps)


    def save_interpolation_dict(self):
        demo_file = open("demo.pkl", "wb")
        pickle.dump(self.interpolated_dict, demo_file)
        demo_file.close()

    def load_interpolation_dict(self):  
        """
            Loads the demo out of a dict, 
            Once its loaded, it will also update the number of interpolations for the number of steps bar
        """
        demo_file = open("demo.pkl", "rb")
        self.interpolated_dict = {}
        self.interpolated_dict = pickle.load(demo_file)
        self.numStepsSpinbox.setProperty("value", len(self.interpolated_dict.keys()))
    

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = DeepSDFLerpGUI()
    gui.show()
    sys.exit(app.exec())