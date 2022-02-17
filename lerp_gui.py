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
                 second_is_file=False, second_mesh_list="", second_mesh_path="",
                 num_steps=1, mesh_array=[]):
        self.first_is_file = first_is_file
        self.first_mesh_list = first_mesh_list
        self.first_mesh_path = first_mesh_path

        self.second_is_file = second_is_file
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
        self.saveDemoButton.clicked.connect(self.save_lerp_demo)
        self.loadDemoButton.clicked.connect(self.load_lerp_demo)
        self.exportCurrentMeshButton.clicked.connect(self.export_current_mesh)

    def save_lerp_demo(self):
        """
        Save lerp demo to file
        """
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save File')
        if path == '':
            return
        with open(path, 'wb') as fp:
            pickle.dump(self.lerp_demo, fp)
    
    def load_lerp_demo(self):
        """
        Load lerp demo from file
        """
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open File')
        if path == '':
            return
        with open(path, 'rb') as fp:
            self.lerp_demo = pickle.load(fp)
        
        # Also populate GUI elements
        if self.lerp_demo.first_is_file:
            self.first_fromList.setChecked(False)
            self.first_fromFile.setChecked(True)
            self.first_pathEdit.setText(self.lerp_demo.first_mesh_path)
        else:
            self.first_fromList.setChecked(True)
            self.first_fromFile.setChecked(False)
            self.fist_comboBox.setCurrentText(self.lerp_demo.first_mesh_list)
        if self.lerp_demo.second_is_file:
            self.second_fromList.setChecked(False)
            self.second_fromFile.setChecked(True)
            self.second_pathEdit.setText(self.lerp_demo.second_mesh_path)
        else:
            self.second_fromList.setChecked(True)
            self.second_fromFile.setChecked(False)
            self.second_comboBox.setCurrentText(self.lerp_demo.second_mesh_list)
        self.numStepsSpinbox.setValue(self.lerp_demo.num_steps)
        self.setup_slider()

        # And finally draw to screen
        self.interpolateSlider.setValue(0)
        self.slider_changed()  # In case the value was already zero

    def export_current_mesh(self):
        """
        Export the current mesh in the view
        """
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save File')
        if path == '':
            return
        if len(self.lerp_demo.mesh_array) == 0:
            return
        mesh = self.lerp_demo.mesh_array[self.interpolateSlider.value()]
        mesh.export(path)


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
        self.currentStepLabel.setText(_translate(
            "DeepSDFLerpGUI", "{}{}".format("Current step: ", curr_step)))

    def interpolate(self):
        """
        Called when "Interpolate" button is pressed.
        Load the corresponding latent vectors based on user selection,
        and compute the mesh interpolation.
        """
        self.lerp_demo = LerpDemo()  # zero the current demo, if exists
        self.setup_slider()

        latent_first = self.load_mesh_latent(0)
        latent_second = self.load_mesh_latent(1)

        # Load trained DeepSDF
        model = DeepSDF().to(device)
        model.load_state_dict(torch.load(
            CHECKPOINT_DEEPSDF, map_location=device))

        mesh_array = []

        # Note that the total overall count of meshes is
        # the number of inbetween steps + start + end
        for i in range(self.get_num_steps() + 2):
            alpha = i / (self.get_num_steps() + 1)
            latent = alpha * latent_second + (1 - alpha) * latent_first
            mesh = utils.deepsdf_to_mesh(
                model, latent, eps=0.05, device=device)
            mesh_array.append(mesh)

        # Update the demo
        self.lerp_demo = LerpDemo(
            self.first_fromFile.isChecked(
            ), self.fist_comboBox.currentText(), self.first_pathEdit.text(),
            self.second_fromFile.isChecked(
            ), self.second_comboBox.currentText(), self.second_pathEdit.text(),
            self.get_num_steps(), mesh_array
        )

        # Force slider update
        self.interpolateSlider.setValue(0)
        self.slider_changed()  # In case the value was already zero

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
            embedding.load_state_dict(torch.load(CHECKPOINT_EMBEDDING, map_location=device))
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

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = DeepSDFLerpGUI()
    gui.show()
    sys.exit(app.exec())
