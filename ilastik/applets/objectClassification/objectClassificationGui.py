from PyQt4.QtGui import *
from PyQt4 import uic
from PyQt4.QtCore import pyqtSignal, pyqtSlot, Qt, QObject

from ilastik.widgets.featureTableWidget import FeatureEntry
from ilastik.widgets.featureDlg import FeatureDlg
from ilastik.applets.objectExtraction.opObjectExtraction import OpRegionFeatures3d
from ilastik.applets.objectExtraction.opObjectExtraction import default_features_key

import os
import numpy
from ilastik.utility import bind
from ilastik.utility.gui import ThreadRouter, threadRouted
from lazyflow.operators import OpSubRegion

import logging
logger = logging.getLogger(__name__)

from ilastik.applets.layerViewer.layerViewerGui import LayerViewerGui
from ilastik.applets.labeling.labelingGui import LabelingGui

import volumina.colortables as colortables
from volumina.api import \
    LazyflowSource, GrayscaleLayer, ColortableLayer, AlphaModulatedLayer, \
    ClickableColortableLayer, LazyflowSinkSource

from volumina.interpreter import ClickInterpreter

def _listReplace(old, new):
    if len(old) > len(new):
        return new + old[len(new):]
    else:
        return new

from ilastik.applets.objectExtraction.objectExtractionGui import FeatureSelectionDialog


class FeatureSubSelectionDialog(FeatureSelectionDialog):
    def __init__(self, featureDict, selectedFeatures=None, parent=None, ndim=3):
        super(FeatureSubSelectionDialog, self).__init__(featureDict, selectedFeatures, parent, ndim)
        self.setObjectName("FeatureSubSelectionDialog")
        self.ui.spinBox_X.setEnabled(False)
        self.ui.spinBox_Y.setEnabled(False)
        self.ui.spinBox_Z.setEnabled(False)
        self.ui.spinBox_X.setVisible(False)
        self.ui.spinBox_Y.setVisible(False)
        self.ui.spinBox_Z.setVisible(False)
        self.ui.marginLabel.setVisible(False)
        self.ui.label.setVisible(False)
        self.ui.label_2.setVisible(False)
        self.ui.label_3.setVisible(False)
        self._setAll(Qt.Checked)


class ObjectClassificationGui(LabelingGui):
    """A subclass of LabelingGui for labeling objects.

    Handles labeling objects, viewing the predicted results, and
    displaying warnings from the top level operator. Also provides a
    dialog for choosing subsets of the precalculated features provided
    by the object extraction applet.

    """

    def centralWidget(self):
        return self

    def appletDrawers(self):
        # Get the labeling drawer from the base class
        labelingDrawer = super(ObjectClassificationGui, self).appletDrawers()[0][1]
        return [("Training", labelingDrawer)]

    def reset(self):
        # Base class first
        super(ObjectClassificationGui, self).reset()

        # Ensure that we are NOT in interactive mode
        self.labelingDrawerUi.checkInteractive.setChecked(False)
        self.labelingDrawerUi.checkShowPredictions.setChecked(False)

    def __init__(self, op, shellRequestSignal, guiControlSignal):
        # Tell our base class which slots to monitor
        labelSlots = LabelingGui.LabelingSlots()
        labelSlots.labelInput = op.LabelInputs
        labelSlots.labelOutput = op.LabelImages

        labelSlots.labelEraserValue = op.Eraser
        labelSlots.labelDelete = op.DeleteLabel

        labelSlots.maxLabelValue = op.NumLabels
        labelSlots.labelsAllowed = op.LabelsAllowedFlags

        # We provide our own UI file (which adds an extra control for
        # interactive mode) This UI file is copied from
        # pixelClassification pipeline
        #
        labelingDrawerUiPath = os.path.split(__file__)[0] + '/labelingDrawer.ui'

        # Base class init
        super(ObjectClassificationGui, self).__init__(labelSlots, op,
                                                      labelingDrawerUiPath,
                                                      crosshair=False)

        self.op = op
        self.guiControlSignal = guiControlSignal
        self.shellRequestSignal = shellRequestSignal

        topLevelOp = self.topLevelOperatorView.viewed_operator()
        self.threadRouter = ThreadRouter(self)
        op.Warnings.notifyDirty(self.handleWarnings)

        # unused
        self.labelingDrawerUi.savePredictionsButton.setEnabled(False)
        self.labelingDrawerUi.savePredictionsButton.setVisible(False)

        self.labelingDrawerUi.brushSizeComboBox.setEnabled(False)
        self.labelingDrawerUi.brushSizeComboBox.setVisible(False)
        
        self.labelingDrawerUi.brushSizeCaption.setVisible(False)


        # button handlers
        self._interactiveMode = False
        self._showPredictions = False
        self._labelMode = True

        self.labelingDrawerUi.subsetFeaturesButton.clicked.connect(
            self.handleSubsetFeaturesClicked)
        self.labelingDrawerUi.checkInteractive.toggled.connect(
            self.handleInteractiveModeClicked)
        self.labelingDrawerUi.checkShowPredictions.toggled.connect(
            self.handleShowPredictionsClicked)

        # enable/disable buttons logic
        self.op.ObjectFeatures.notifyDirty(bind(self.checkEnableButtons))
        self.op.NumLabels.notifyDirty(bind(self.checkEnableButtons))
        self.op.SelectedFeatures.notifyDirty(bind(self.checkEnableButtons))
        self.checkEnableButtons()

    @property
    def labelMode(self):
        return self._labelMode

    @labelMode.setter
    def labelMode(self, val):
        self.labelingDrawerUi.labelListView.allowDelete = val
        self.labelingDrawerUi.AddLabelButton.setEnabled(val)
        self._labelMode = val

    @property
    def interactiveMode(self):
        return self._interactiveMode

    @interactiveMode.setter
    def interactiveMode(self, val):
        logger.debug("setting interactive mode to '%r'" % val)
        self._interactiveMode = val
        self.labelingDrawerUi.checkInteractive.setChecked(val)
        if val:
            self.showPredictions = True
        self.labelMode = not val

    @pyqtSlot()
    def handleInteractiveModeClicked(self):
        self.interactiveMode = self.labelingDrawerUi.checkInteractive.isChecked()

    @property
    def showPredictions(self):
        return self._showPredictions

    @showPredictions.setter
    def showPredictions(self, val):
        self._showPredictions = val
        self.labelingDrawerUi.checkShowPredictions.setChecked(val)
        for layer in self.layerstack:
            if "Prediction" in layer.name:
                layer.visible = val

        if self.labelMode and not val:
            self.labelMode = False
            # And hide all segmentation layers
            for layer in self.layerstack:
                if "Segmentation" in layer.name:
                    layer.visible = False

    @pyqtSlot()
    def handleShowPredictionsClicked(self):
        self.showPredictions = self.labelingDrawerUi.checkShowPredictions.isChecked()

    @pyqtSlot()
    def handleSubsetFeaturesClicked(self):
        mainOperator = self.topLevelOperatorView
        computedFeatures = mainOperator.ComputedFeatureNames([]).wait()
        if mainOperator.SelectedFeatures.ready():
            selectedFeatures = mainOperator.SelectedFeatures([]).wait()
        else:
            selectedFeatures = None

        ndim = 3 # FIXME
        dlg = FeatureSubSelectionDialog(computedFeatures,
                                        selectedFeatures=selectedFeatures, ndim=ndim)
        dlg.exec_()
        if dlg.result() == QDialog.Accepted:
            if len(dlg.selectedFeatures) == 0:
                self.interactiveMode = False
            mainOperator.SelectedFeatures.setValue(dlg.selectedFeatures)

    @pyqtSlot()
    def checkEnableButtons(self):
        feats_enabled = True
        predict_enabled = True
        labels_enabled = True

        if self.op.ComputedFeatureNames.ready():
            featnames = self.op.ComputedFeatureNames([]).wait()
            if len(featnames) == 0:
                feats_enabled = False
        else:
            feats_enabled = False

        if feats_enabled:
            if self.op.SelectedFeatures.ready():
                featnames = self.op.SelectedFeatures([]).wait()
                if len(featnames) == 0:
                    predict_enabled = False
            else:
                predict_enabled = False

            if self.op.NumLabels.ready():
                if self.op.NumLabels.value < 2:
                    predict_enabled = False
            else:
                predict_enabled = False
        else:
            predict_enabled = False

        if not predict_enabled:
            self.interactiveMode = False
            self.showPredictions = False

        self.labelingDrawerUi.subsetFeaturesButton.setEnabled(feats_enabled)
        self.labelingDrawerUi.checkInteractive.setEnabled(predict_enabled)
        self.labelingDrawerUi.checkShowPredictions.setEnabled(predict_enabled)
        self.labelingDrawerUi.AddLabelButton.setEnabled(labels_enabled)


    def initAppletDrawerUi(self):
        """
        Load the ui file for the applet drawer, which we own.
        """
        localDir = os.path.split(__file__)[0]

        # We don't pass self here because we keep the drawer ui in a
        # separate object.
        self.drawer = uic.loadUi(localDir+"/drawer.ui")

    ### Function dealing with label name and color consistency
    def _getNext(self, slot, parentFun, transform=None):
        numLabels = self.labelListData.rowCount()
        value = slot.value
        if numLabels < len(value):
            result = value[numLabels]
            if transform is not None:
                result = transform(result)
            return result
        else:
            return parentFun()

    def _onLabelChanged(self, parentFun, mapf, slot):
        parentFun()
        new = map(mapf, self.labelListData)
        old = slot.value
        slot.setValue(_listReplace(old, new))

    def getNextLabelName(self):
        return self._getNext(self.topLevelOperatorView.LabelNames,
                             super(ObjectClassificationGui, self).getNextLabelName)

    def getNextLabelColor(self):
        return self._getNext(
            self.topLevelOperatorView.LabelColors,
            super(ObjectClassificationGui, self).getNextLabelColor,
            lambda x: QColor(*x)
        )

    def getNextPmapColor(self):
        return self._getNext(
            self.topLevelOperatorView.PmapColors,
            super(ObjectClassificationGui, self).getNextPmapColor,
            lambda x: QColor(*x)
        )

    def onLabelNameChanged(self):
        self._onLabelChanged(super(ObjectClassificationGui, self).onLabelNameChanged,
                             lambda l: l.name,
                             self.topLevelOperatorView.LabelNames)

    def onLabelColorChanged(self):
        self._onLabelChanged(super(ObjectClassificationGui, self).onLabelColorChanged,
                             lambda l: (l.brushColor().red(),
                                        l.brushColor().green(),
                                        l.brushColor().blue()),
                             self.topLevelOperatorView.LabelColors)


    def onPmapColorChanged(self):
        self._onLabelChanged(super(ObjectClassificationGui, self).onPmapColorChanged,
                             lambda l: (l.pmapColor().red(),
                                        l.pmapColor().green(),
                                        l.pmapColor().blue()),
                             self.topLevelOperatorView.PmapColors)

    def _onLabelRemoved(self, parent, start, end):
        super(ObjectClassificationGui, self)._onLabelRemoved(parent, start, end)
        op = self.topLevelOperatorView
        op.removeLabel(start)
        for slot in (op.LabelNames, op.LabelColors, op.PmapColors):
            value = slot.value
            value.pop(start)
            slot.setValue(value)


    def createLabelLayer(self, direct=False):
        """Return a colortable layer that displays the label slot
        data, along with its associated label source.

        direct: whether this layer is drawn synchronously by volumina

        """
        labelInput = self._labelingSlots.labelInput
        labelOutput = self._labelingSlots.labelOutput

        if not labelOutput.ready():
            return (None, None)
        else:
            self._colorTable16[15] = QColor(Qt.black).rgba() #for the objects with NaNs in features


            labelsrc = LazyflowSinkSource(labelOutput,
                                          labelInput)
            labellayer = ColortableLayer(labelsrc,
                                         colorTable=self._colorTable16,
                                         direct=direct)

            labellayer.segmentationImageSlot = self.op.SegmentationImagesOut
            labellayer.name = "Labels"
            labellayer.ref_object = None
            labellayer.zeroIsTransparent  = False
            labellayer.colortableIsRandom = True

            clickInt = ClickInterpreter(self.editor, labellayer,
                                        self.onClick, right=False,
                                        double=False)
            self.editor.brushingInterpreter = clickInt

            return labellayer, labelsrc

    def setupLayers(self):

        # Base class provides the label layer.
        layers = super(ObjectClassificationGui, self).setupLayers()

        labelOutput = self._labelingSlots.labelOutput
        binarySlot = self.op.BinaryImages
        segmentedSlot = self.op.SegmentationImages
        rawSlot = self.op.RawImages

        if segmentedSlot.ready():
            ct = colortables.create_default_16bit()
            self.objectssrc = LazyflowSource(segmentedSlot)
            ct[0] = QColor(0, 0, 0, 0).rgba() # make 0 transparent
            layer = ColortableLayer(self.objectssrc, ct)
            layer.name = "Objects"
            layer.opacity = 0.5
            layer.visible = True
            layers.append(layer)

        if binarySlot.ready():
            ct_binary = [QColor(0, 0, 0, 0).rgba(),
                         QColor(255, 255, 255, 255).rgba()]
            self.binaryimagesrc = LazyflowSource(binarySlot)
            layer = ColortableLayer(self.binaryimagesrc, ct_binary)
            layer.name = "Binary Image"
            layer.visible = False
            layers.append(layer)

        #This is just for colors
        labels = self.labelListData
        for channel, probSlot in enumerate(self.op.PredictionProbabilityChannels):
            if probSlot.ready() and channel < len(labels):
                ref_label = labels[channel]
                probsrc = LazyflowSource(probSlot)
                probLayer = AlphaModulatedLayer( probsrc,
                                                 tintColor=ref_label.pmapColor(),
                                                 range=(0.0, 1.0),
                                                 normalize=(0.0, 1.0) )
                probLayer.opacity = 0.25
                probLayer.visible = self.labelingDrawerUi.checkInteractive.isChecked()

                def setLayerColor(c, predictLayer=probLayer):
                    predictLayer.tintColor = c

                def setLayerName(n, predictLayer=probLayer):
                    newName = "Prediction for %s" % n
                    predictLayer.name = newName

                setLayerName(ref_label.name)
                ref_label.pmapColorChanged.connect(setLayerColor)
                ref_label.nameChanged.connect(setLayerName)
                layers.insert(0, probLayer)

        predictionSlot = self.op.PredictionImages
        if predictionSlot.ready():
            self.predictsrc = LazyflowSource(predictionSlot)
            self.predictlayer = ColortableLayer(self.predictsrc,
                                                colorTable=self._colorTable16)
            self.predictlayer.name = "Prediction"
            self.predictlayer.ref_object = None
            self.predictlayer.visible = self.labelingDrawerUi.checkInteractive.isChecked()

            # put first, so that it is visible after hitting "live
            # predict".
            layers.insert(0, self.predictlayer)

        badObjectsSlot = self.op.BadObjectImages
        if badObjectsSlot.ready():
            ct_black = [0, QColor(Qt.black).rgba()]
            self.badSrc = LazyflowSource(badObjectsSlot)
            self.badLayer = ColortableLayer(self.badSrc, colorTable = ct_black)
            self.badLayer.name = "Ambiguous objects"
            self.badLayer.visible = False
            layers.append(self.badLayer)

        if rawSlot.ready():
            self.rawimagesrc = LazyflowSource(rawSlot)
            layer = self.createStandardLayerFromSlot(rawSlot)
            layer.name = "Raw data"
            layers.append(layer)

        # since we start with existing labels, it makes sense to start
        # with the first one selected. This would make more sense in
        # __init__(), but it does not take effect there.
        #self.selectLabel(0)

        return layers

    @staticmethod
    def _getObject(slot, pos5d):
        slicing = tuple(slice(i, i+1) for i in pos5d)
        arr = slot[slicing].wait()
        return arr.flat[0]

    def onClick(self, layer, pos5d, pos):
        """Extracts the object index that was clicked on and updates
        that object's label.

        """
        label = self.editor.brushingModel.drawnNumber
        if label == self.editor.brushingModel.erasingNumber:
            label = 0

        topLevelOp = self.topLevelOperatorView.viewed_operator()
        imageIndex = topLevelOp.LabelInputs.index( self.topLevelOperatorView.LabelInputs )

        operatorAxisOrder = self.topLevelOperatorView.SegmentationImagesOut.meta.getAxisKeys()
        assert operatorAxisOrder == list('txyzc'), \
            "Need to update onClick() if the operator no longer expects volumina axis order.  Operator wants: {}".format( operatorAxisOrder )
        self.topLevelOperatorView.assignObjectLabel(imageIndex, pos5d, label)

    def handleEditorRightClick(self, position5d, globalWindowCoordinate):
        layer = self.getLayer('Labels')
        obj = self._getObject(layer.segmentationImageSlot, position5d)
        if obj == 0:
            return

        menu = QMenu(self)
        text = "print info for object {}".format(obj)
        menu.addAction(text)
        action = menu.exec_(globalWindowCoordinate)
        if action is not None and action.text() == text:
            numpy.set_printoptions(precision=4)
            print "------------------------------------------------------------"
            print "object:         {}".format(obj)
            
            t = position5d[0]
            labels = self.op.LabelInputs([t]).wait()[t]
            if len(labels) > obj:
                label = int(labels[obj])
            else:
                label = "none"
            print "label:          {}".format(label)
            
            print 'features:'
            feats = self.op.ObjectFeatures([t]).wait()[t]
            selected = self.op.SelectedFeatures([]).wait()
            for plugin in sorted(feats.keys()):
                if plugin == default_features_key or plugin not in selected:
                    continue
                print "Feature category: {}".format(plugin)
                for featname in sorted(feats[plugin].keys()):
                    if featname not in selected[plugin]:
                        continue
                    value = feats[plugin][featname]
                    ft = numpy.asarray(value.squeeze())[obj]
                    print "{}: {}".format(featname, ft)

            if len(selected)>0 and label!='none':
                if self.op.Predictions.ready():
                    preds = self.op.Predictions([t]).wait()[t]
                    if len(preds) >= obj:
                        pred = int(preds[obj])
                else:
                    pred = 'none'
    
                if self.op.Probabilities.ready():
                    probs = self.op.Probabilities([t]).wait()[t]
                    if len(probs) >= obj:
                        prob = probs[obj]
                else:
                    prob = 'none'
    
                print "probabilities:  {}".format(prob)
                print "prediction:     {}".format(pred)

            
            print "------------------------------------------------------------"

    def setVisible(self, visible):
        super(ObjectClassificationGui, self).setVisible(visible)

        if visible:
            temp = self.op.triggerTransferLabels(self.op.current_view_index())
        else:
            temp = None
        if temp is not None:
            new_labels, old_labels_lost, new_labels_lost = temp
            labels_lost = dict(old_labels_lost.items() + new_labels_lost.items())
            if sum(len(v) for v in labels_lost.itervalues()) > 0:
                self.warnLost(labels_lost)

    def warnLost(self, labels_lost):
        box = QMessageBox(QMessageBox.Warning,
                          'Warning',
                          'Some of your labels could not be transferred',
                          QMessageBox.NoButton,
                          self)
        messages = {
            'full': "These labels were lost completely:",
            'partial': "These labels were lost partially:",
            'conflict': "These new labels conflicted:"
        }
        default_message = "These labels could not be transferred:"

        _sep = "\t"
        cases = []
        for k, val in labels_lost.iteritems():
            if len(val) > 0:
                msg = messages.get(k, default_message)
                axis = _sep.join(["X", "Y", "Z"])
                coords = "\n".join([_sep.join(["{:<8.1f}".format(i) for i in item])
                                    for item in val])
                cases.append("\n".join([msg, axis, coords]))
        box.setDetailedText("\n\n".join(cases))
        box.show()


    @threadRouted
    def handleWarnings(self, *args, **kwargs):
        # FIXME: dialog should not steal focus
        warning = self.op.Warnings[:].wait()
        try:
            box = self.badObjectBox
        except AttributeError:
            box = QMessageBox(QMessageBox.Warning,
                              warning['title'],
                              warning['text'],
                              QMessageBox.NoButton,
                              self)
            box.setWindowModality(Qt.NonModal)
            box.move(self.geometry().width(), 0)
        box.setWindowTitle(warning['title'])
        box.setText(warning['text'])
        box.setInformativeText(warning.get('info', ''))
        box.setDetailedText(warning.get('details', ''))
        box.show()
        self.badObjectBox = box
