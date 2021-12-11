import sys

import openmesh as om
from PyQt5 import QtCore, QtWidgets

from gui.gui import Ui_DeepSDFLerpGUI
from gui.mgl_qt_meshviewer import QGLControllerWidget

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

        self.pushButton_4.clicked.connect(self.show_ted)
    
    def show_ted(self):
        mesh = om.read_trimesh('data/raw_meshes/ted.obj')
        self.gl.set_mesh(mesh)
        

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = DeepSDFLerpGUI()
    gui.show()
    sys.exit(app.exec())