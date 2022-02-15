import os
import sys
import pickle

import torch
from PyQt5 import QtCore, QtWidgets

from gui.gui import Ui_DeepSDFLerpGUI
from gui.mgl_qt_meshviewer import QGLControllerWidget

import utils
from data import *
from model import DeepSDF, LatentEmbedding

import random

# Setup CUDA if available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


class LerpDemo(object):
    """
    Object that encapsulates a single demo of the LerpGUI program
    Contains metadata (selected meshes, number of steps) and interpolated mesh data
    """
    def __init__(self, 
        first_is_file=False, first_mesh_list="", first_mesh_path="", 
        seocnd_is_file=False, second_mesh_list="", second_mesh_path="", 
        num_steps=1, mesh_array=[]):
        self.first_is_file = first_is_file
        self.first_mesh_list = first_mesh_list
        self.first_mesh_path = first_mesh_path

        self.seocnd_is_file = seocnd_is_file
        self.second_mesh_list = second_mesh_list
        self.second_mesh_path = second_mesh_path

        self.num_steps = num_steps
        self.mesh_array = mesh_array


class DeepSDFLerpGUI(QtWidgets.QMainWindow, Ui_DeepSDFLerpGUI):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        #####################
        # Setup OpenGL View
        #####################
        self.gl = QGLControllerWidget(self)
        self.glFrameLayout.addWidget(self.gl)

        timer = QtCore.QTimer(self)
        timer.setInterval(20)  # period, in milliseconds
        timer.timeout.connect(self.gl.updateGL)
        timer.start()

        #######################
        # Setup Mesh Selection 
        #######################
        self.first_fromFile.setChecked(False)
        self.first_fromList.setChecked(True)
        self.second_fromFile.setChecked(False)
        self.second_fromList.setChecked(True)
        self.populate_lists()
        self.first_browseButton.clicked.connect(lambda: self.browse_button(0))
        self.second_browseButton.clicked.connect(lambda: self.browse_button(1))

        
        ###########################
        # Setup Mesh Interpolation 
        ###########################
        self.lerp_demo = LerpDemo()
        self.setup_slider()
        self.lerpButton.clicked.connect(self.interpolate)
        self.interpolateSlider.valueChanged.connect(self.slider_changed)


        # self.label_4.clicked.connect(self.interpulation_press)
        self.saveDemoButton.clicked.connect(self.save_interpolation_dict)
        self.loadDemoButton.clicked.connect(self.load_interpolation_dict)
        self.numStepsSpinbox.valueChanged.connect(self.changed_value)  
        self.alpha = 0.0
        self.curr_step = 0
        self.nm_of_steps = 1
        self.step_size = float(1/self.nm_of_steps)
        self.interpolated_dict = {}

        self.choose_random_meshes = True
    
    def slider_changed(self):
        """
        Called when user moves the slider.
        Update the viewed mesh based on the value.
        """
        if len(self.lerp_demo.mesh_array) == 0:
            return
        
        val = self.interpolateSlider.value()
        mesh = self.lerp_demo.mesh_array[val]
        self.gl.set_mesh(mesh)

        # Update the label
        _translate = QtCore.QCoreApplication.translate
        curr_step = str(val)
        if val == 0:
            curr_step = "First mesh"
        elif val == self.lerp_demo.num_steps + 1:
            curr_step = "Second mesh"
        self.currentStepLabel.setText(_translate("DeepSDFLerpGUI", "{}{}".format("Current step: ", curr_step)))
        

    def interpolate(self):
        """
        Called when "Interpolate" button is pressed.
        Load the corresponding latent vectors based on user selection,
        and compute the mesh interpolation.
        """
        self.lerp_demo = LerpDemo() # zero the current demo, if exists
        self.setup_slider()

        latent_first = self.load_mesh_latent(0)
        latent_second = self.load_mesh_latent(1)

        # Load trained DeepSDF
        model = DeepSDF().to(device)
        model.load_state_dict(torch.load(CHECKPOINT_DEEPSDF, map_location=device))

        mesh_array = []

        # Note that the total overall count of meshes is
        # the number of inbetween steps + start + end
        for i in range(self.get_num_steps() + 2):
            alpha = i / (self.get_num_steps() + 1)
            latent = alpha * latent_second + (1 - alpha) * latent_first
            mesh = utils.deepsdf_to_mesh(model, latent, eps=0.05, device=device)
            mesh_array.append(mesh)

        # Update the demo
        self.lerp_demo = LerpDemo(
            self.first_fromFile.isChecked(), self.fist_comboBox.currentText(), self.first_pathEdit.text(),
            self.second_fromFile.isChecked(), self.second_comboBox.currentText(), self.second_pathEdit.text(),
            self.get_num_steps(), mesh_array
        )

        # Force slider update
        self.interpolateSlider.setValue(0)
        self.slider_changed() # In case the value was already zero


    def load_mesh_latent(self, idx):
        """
        Given the selected option (list or file), return the corresponding
        latent vector. If list then we use the embedding network, otherwise optimize
        the latent vector based on samples.
        Idx = 0 for first mesh, otherwise second
        """
        if idx == 0:
            is_from_file = self.first_fromFile.isChecked()
            file_path = self.first_pathEdit.text()
            list_item = self.fist_comboBox.currentText()
        else:
            is_from_file = self.second_fromFile.isChecked()
            file_path = self.second_pathEdit.text()
            list_item = self.second_comboBox.currentText()
        
        if is_from_file:
            return utils.mesh_to_deepSDF(file_path)
        else:
            embedding = LatentEmbedding(NUM_MESHES)
            embedding.load_state_dict(torch.load(CHECKPOINT_EMBEDDING))
            return embedding(torch.tensor(MESHES[list_item])).detach()

    def get_num_steps(self):
        """
        Get the number of steps user selected
        """
        return int(self.numStepsSpinbox.value())

    def setup_slider(self):
        """
        Setup the tick interval on the slider
        """
        self.interpolateSlider.setMinimum(0)
        self.interpolateSlider.setMaximum(self.get_num_steps() + 1)
        self.interpolateSlider.setTickInterval(1)
        self.interpolateSlider.setSingleStep(1)



    def browse_button(self, idx):
        """
        Select a path for a given mesh
        """
        if idx == 0:
            line_edit = self.first_pathEdit
            check_from_file = self.first_fromFile
            check_from_list = self.first_fromList
        else:
            line_edit = self.second_pathEdit
            check_from_file = self.second_fromFile
            check_from_list = self.second_fromList
        
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Load File')
        if path == '':
            return
        line_edit.setText(path)
        check_from_list.setChecked(False)
        check_from_file.setChecked(True)


    
    def populate_lists(self):
        """
        Add all mesh options to the item combo boxes
        """
        for mesh in MESHES:
            self.fist_comboBox.addItem(mesh)
            self.second_comboBox.addItem(mesh)

        
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