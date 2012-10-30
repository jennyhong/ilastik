'''
Created on Oct 14, 2012

@author: fredo
'''
from ilastik.applets.layerViewer.layerViewerGui import LayerViewerGui
from volumina.pixelpipeline.imagepump import ImagePump
from volumina.slicingtools import SliceProjection, SliceProjectionTest
from volumina.eventswitch import EventSwitch
from volumina.navigationControler import NavigationControler, NavigationInterpreter
from volumina.brushingcontroler import BrushingControler, BrushingInterpreter, CrosshairControler
from volumina.clickReportingInterpreter import ClickReportingInterpreter
from PyQt4.QtGui import QTransform,QWidget
from PyQt4 import uic
import os
useVTK = True

import logging

try:
    from volumina.view3d.view3d import OverviewScene
except:
    print "Warning: could not import optional dependency VTK"
    useVTK = False

class LightfieldGui(LayerViewerGui):
    
    APPLET_DRAWER_PATH = os.path.join(os.path.dirname(__file__),"drawer.ui")
    logger = logging.getLogger(__name__)
    
    def __init__(self, toplevelOperator):
        super(LightfieldGui,self).__init__(toplevelOperator)
        #=======================================================================
        # rearrange views
        #=======================================================================
        x_slicing_view = self.volumeEditorWidget.quadview.splitHorizontal1.widget(1)
        y_slicing_view = self.volumeEditorWidget.quadview.splitHorizontal2.widget(0)
        z_slicing_view = self.volumeEditorWidget.quadview.splitHorizontal1.widget(0)
        view_3d = self.volumeEditorWidget.quadview.splitHorizontal2.widget(1)
        
        self.volumeEditorWidget.quadview.splitHorizontal1.addWidget(x_slicing_view)
        self.volumeEditorWidget.quadview.splitHorizontal1.addWidget(y_slicing_view)
        self.volumeEditorWidget.quadview.splitHorizontal1.addWidget(z_slicing_view)
        self.volumeEditorWidget.quadview.splitHorizontal2.addWidget(view_3d)
        
        self.initDrawers()
        
        
    def initDrawers(self):
        self._drawers = uic.loadUi(self.APPLET_DRAWER_PATH)
        
        self._drawers.editChannelSubmit.clicked.connect(self.editChannel)
        self._drawers.editGaussSubmit.clicked.connect(self.editGauss)
        self._drawers.editContrastSubmit.clicked.connect(self.editContrast)
        self._drawers.editGammaSubmit.clicked.connect(self.editGamma)
        self._drawers.editMedianSubmit.clicked.connect(self.editMedian)
        
    def appletDrawers(self):
        return [("Lightfield View", self._drawers )]
    
    def editGauss(self):
        self.logger.info("Edit Gauss has been clicked.")
        radius = self._drawers.editGaussRadius.value()
        self.operation = "Gauss"
        self.options = {"radius":radius}
        
        
    def editChannel(self):
        self.logger.info("Edit channel has been clicked")
        r = self._drawers.editChannelR.value()
        b = self._drawers.editChannelB.value()
        g = self._drawers.editChannelG.value()
        self.operation = "Channel"
        self.options = {"red": r,"green": g,"blue":b}
        
    def editContrast(self):
        contrast = self._drawers.editContrast.value()
        brightness = self._drawers.editBrightness.value()
        self.operation="Contrast"
        self.options = {"brightness" : brightness, "contrast" : contrast}
        
    def editGamma(self):
        gamma = self._drawers.editGamma.value()
        self.operation="Gamma"
        self.options = {"gamma" : gamma}
        
    def editMedian(self):
        size = self._drawers.editMedianSize.value()
        self.operation = "Median"
        self.options = {"size": size}
        
    @property
    def operation(self):
        pass
    
    @operation.setter
    def operation(self,value):
        self.topLevelOperator.Operation.setValue(value)
    
    @property
    def options(self):
        pass
    
    @options.setter
    def options(self,value):
        self.topLevelOperator.Options.setValue(value)
        
    def setupLayers(self, currentImageIndex):
        layers = []

        # Show the thresholded data
        outputImageSlot = self.topLevelOperator.Output[ currentImageIndex ]
        if outputImageSlot.ready():
            outputLayer = self.createStandardLayerFromSlot( outputImageSlot )
            outputLayer.name = "min <= x <= max"
            outputLayer.visible = True
            outputLayer.opacity = 0.75
            layers.append(outputLayer)
        
#        # Show the  data
#        invertedOutputSlot = self.topLevelOperator.InvertedOutput[ currentImageIndex ]
#        if invertedOutputSlot.ready():
#            invertedLayer = self.createStandardLayerFromSlot( invertedOutputSlot )
#            invertedLayer.name = "(x < min) U (x > max)"
#            invertedLayer.visible = True
#            invertedLayer.opacity = 0.25
#            layers.append(invertedLayer)
        
        # Show the raw input data
        inputImageSlot = self.topLevelOperator.InputImage[ currentImageIndex ]
        if inputImageSlot.ready():
            inputLayer = self.createStandardLayerFromSlot( inputImageSlot )
            inputLayer.name = "Raw Input"
            inputLayer.visible = True
            inputLayer.opacity = 1.0
            layers.append(inputLayer)

        return layers
    
        