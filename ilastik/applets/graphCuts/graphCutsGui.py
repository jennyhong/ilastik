from PyQt4.QtGui import *
from PyQt4 import uic

import os

from ilastik.applets.layerViewer.layerViewerGui import LayerViewerGui
from volumina.widgets.thresholdingWidget import ThresholdingWidget

import logging
logger = logging.getLogger(__name__)
traceLogger = logging.getLogger('TRACE.' + __name__)
from lazyflow.utility import Tracer

from ilastik.utility import bind

class GraphCutsGui(LabelingGui):
    """
    """
    
    ###########################################
    ### AppletGuiInterface Concrete Methods ###
    ###########################################

    def appletDrawer(self):
        return self.getAppletDrawerUi()

    # (Other methods already provided by our base class)

    ###########################################
    ###########################################
    
    def __init__(self, topLevelOperatorView):
        """
        """
        with Tracer(traceLogger):
            self.topLevelOperatorView = topLevelOperatorView
            super(GraphCutsGui, self).__init__(self.topLevelOperatorView)

        # Tell our base class which slots to monitor
        labelSlots = LabelingGui.LabelingSlots()
        labelSlots.labelInput = topLevelOperatorView.LabelInputs
        labelSlots.labelOutput = topLevelOperatorView.LabelImages
        labelSlots.labelEraserValue = topLevelOperatorView.opLabelPipeline.opLabelArray.eraser
        labelSlots.labelDelete = topLevelOperatorView.opLabelPipeline.opLabelArray.deleteLabel

        # Base class init
        super(PixelClassificationGui, self).__init__( labelSlots, topLevelOperatorView, labelingDrawerUiPath )
        
        self.topLevelOperatorView = topLevelOperatorView
        self.interactiveModeActive = False
        self._currentlySavingPredictions = False

        self.labelingDrawerUi.savePredictionsButton.clicked.connect(self.onSavePredictionsButtonClicked)
        self.labelingDrawerUi.savePredictionsButton.setIcon( QIcon(ilastikIcons.Save) )
        
        self.labelingDrawerUi.liveUpdateButton.setEnabled(False)
        self.labelingDrawerUi.liveUpdateButton.setIcon( QIcon(ilastikIcons.Play) )
        self.labelingDrawerUi.liveUpdateButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.labelingDrawerUi.liveUpdateButton.toggled.connect( self.toggleInteractive )

        self.topLevelOperatorView.MaxLabelValue.notifyDirty( bind(self.handleLabelSelectionChange) )
        
        self._initShortcuts()

        try:
            self.render = True
            self._renderedLayers = {} # (layer name, label number)
            self._renderMgr = RenderingManager(
                renderer=self.editor.view3d.qvtk.renderer,
                qvtk=self.editor.view3d.qvtk)
        except:
            self.render = False
            
    def initAppletDrawerUi(self):
        with Tracer(traceLogger):
            # Load the ui file (find it in our own directory)
            localDir = os.path.split(__file__)[0]
            self._drawer = uic.loadUi(localDir+"/drawer.ui")
            
            layout = QVBoxLayout()
            layout.setSpacing(0)
            self._drawer.setLayout( layout )
    
            thresholdWidget = ThresholdingWidget(self)
            thresholdWidget.valueChanged.connect( self.handleThresholdGuiValuesChanged )
            self._drawer.layout().addWidget( thresholdWidget )
            self._drawer.layout().addSpacerItem( QSpacerItem(0,0,vPolicy=QSizePolicy.Expanding) )
            
            def updateDrawerFromOperator():
                minValue, maxValue = (0,255)

                if self.topLevelOperatorView.MinValue.ready():
                    minValue = self.topLevelOperatorView.MinValue.value
                if self.topLevelOperatorView.MaxValue.ready():
                    maxValue = self.topLevelOperatorView.MaxValue.value

                thresholdWidget.setValue(minValue, maxValue)
                
            self.topLevelOperatorView.MinValue.notifyDirty( bind(updateDrawerFromOperator) )
            self.topLevelOperatorView.MaxValue.notifyDirty( bind(updateDrawerFromOperator) )
            updateDrawerFromOperator()
            
    def handleThresholdGuiValuesChanged(self, minVal, maxVal):
        with Tracer(traceLogger):
            self.topLevelOperatorView.MinValue.setValue(minVal)
            self.topLevelOperatorView.MaxValue.setValue(maxVal)
    
    def getAppletDrawerUi(self):
        return self._drawer
    
    def setupLayers(self):
        """
        Called by our base class when one of our data slots has changed.
        This function creates a layer for each slot we want displayed in the volume editor.
        """
        # Base class provides the label layer.
        layers = super(PixelClassificationGui, self).setupLayers()
        labels = self.labelListData

        # Add each of the segmentations
        for channel, segmentationSlot in enumerate(self.topLevelOperatorView.SegmentationChannels):
            if segmentationSlot.ready() and channel < len(labels):
                ref_label = labels[channel]
                segsrc = LazyflowSource(segmentationSlot)
                segLayer = AlphaModulatedLayer( segsrc,
                                                tintColor=ref_label.pmapColor(),
                                                range=(0.0, 1.0),
                                                normalize=(0.0, 1.0) )

                segLayer.opacity = 1
                segLayer.visible = self.labelingDrawerUi.liveUpdateButton.isChecked()
                segLayer.visibleChanged.connect(self.updateShowSegmentationCheckbox)

                def setLayerColor(c, segLayer=segLayer):
                    segLayer.tintColor = c
                    self._update_rendering()

                def setSegLayerName(n, segLayer=segLayer):
                    oldname = segLayer.name
                    newName = "Segmentation (%s)" % n
                    segLayer.name = newName
                    if not self.render:
                        return
                    if oldname in self._renderedLayers:
                        label = self._renderedLayers.pop(oldname)
                        self._renderedLayers[newName] = label

                setSegLayerName(ref_label.name)

                ref_label.pmapColorChanged.connect(setLayerColor)
                ref_label.nameChanged.connect(setSegLayerName)
                #check if layer is 3d before adding the "Toggle 3D" option
                #this check is done this way to match the VolumeRenderer, in
                #case different 3d-axistags should be rendered like t-x-y
                #_axiskeys = segmentationSlot.meta.getAxisKeys()
                if len(segmentationSlot.meta.shape) == 4:
                    #the Renderer will cut out the last shape-dimension, so
                    #we're checking for 4 dimensions
                    self._setup_contexts(segLayer)
                layers.append(segLayer)
        
        # Add each of the predictions
        for channel, predictionSlot in enumerate(self.topLevelOperatorView.PredictionProbabilityChannels):
            if predictionSlot.ready() and channel < len(labels):
                ref_label = labels[channel]
                predictsrc = LazyflowSource(predictionSlot)
                predictLayer = AlphaModulatedLayer( predictsrc,
                                                    tintColor=ref_label.pmapColor(),
                                                    range=(0.0, 1.0),
                                                    normalize=(0.0, 1.0) )
                predictLayer.opacity = 0.25
                predictLayer.visible = self.labelingDrawerUi.liveUpdateButton.isChecked()
                predictLayer.visibleChanged.connect(self.updateShowPredictionCheckbox)

                def setLayerColor(c, predictLayer=predictLayer):
                    predictLayer.tintColor = c

                def setPredLayerName(n, predictLayer=predictLayer):
                    newName = "Prediction for %s" % n
                    predictLayer.name = newName

                setPredLayerName(ref_label.name)
                ref_label.pmapColorChanged.connect(setLayerColor)
                ref_label.nameChanged.connect(setPredLayerName)
                layers.append(predictLayer)

        # Add the raw data last (on the bottom)
        inputDataSlot = self.topLevelOperatorView.InputImages
        if inputDataSlot.ready():
            inputLayer = self.createStandardLayerFromSlot( inputDataSlot )
            inputLayer.name = "Input Data"
            inputLayer.visible = True
            inputLayer.opacity = 1.0

            def toggleTopToBottom():
                index = self.layerstack.layerIndex( inputLayer )
                self.layerstack.selectRow( index )
                if index == 0:
                    self.layerstack.moveSelectedToBottom()
                else:
                    self.layerstack.moveSelectedToTop()

            inputLayer.shortcutRegistration = (
                "Prediction Layers",
                "Bring Input To Top/Bottom",
                QShortcut( QKeySequence("i"), self.viewerControlWidget(), toggleTopToBottom),
                inputLayer )
            layers.append(inputLayer)
        
        self.handleLabelSelectionChange()
        return layers