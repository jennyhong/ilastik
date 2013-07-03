import os
import fnmatch
import glob

from PyQt4.QtGui import QDialog, QFileDialog
from PyQt4 import uic
from ilastik.config import cfg as ilastik_config

class MassFileLoader(QDialog):

    def __init__(self, parent=None, defaultDirectory=None):
        QDialog.__init__(self, parent)
        self.defaultDirectory = defaultDirectory
        self.setWindowTitle("Mass file loader")
        ui_class, widget_class = uic.loadUiType(os.path.split(__file__)[0] + "/massFileLoader.ui")
        self.ui = ui_class()
        self.ui.setupUi(self)
        self.ui.directoryButton.clicked.connect(self.handleDirectoryButtonClicked)
        self.ui.buttonBox.accepted.connect(self.accept)
        self.ui.buttonBox.rejected.connect(self.reject)

    def accept(self):
        QDialog.accept(self)
        pattern = str(self.ui.patternInput.text())
        directory = os.path.expanduser(str(self.ui.directoryInput.text()))
        recursive = self.ui.recursiveBox.isChecked()
        if recursive:
            self.filenames = []
            for root, dirnames, filenames in os.walk(directory):
                self.filenames.extend(os.path.join(root, f).replace('\\', '/')
                                      for f in fnmatch.filter(filenames, pattern))
        else:
            self.filenames = [k.replace('\\', '/') for k in glob.glob(os.path.join(directory, pattern))]
        
        self.filenames = sorted( self.filenames )

    def handleDirectoryButtonClicked(self):
        options = QFileDialog.Options()
        if ilastik_config.getboolean("ilastik", "debug"):
            options |=  QFileDialog.DontUseNativeDialog
        directoryName = QFileDialog.getExistingDirectory(self,
                                                         "Base Directory",
                                                         self.defaultDirectory,
                                                         options=options)
        self.ui.directoryInput.setText(directoryName)
