from lazyflow.graph import Operator, InputSlot, OutputSlot

# Python
import logging
logger = logging.getLogger(__name__)
traceLogger = logging.getLogger("TRACE." + __name__)

# additional modules to install
import sklearn.mixture
import numpy
import pygco

class OpGraphCuts(Operator):
    name = "OpGraphCuts"
    category = "Pointwise"

    InputImage = InputSlot()
    Labels = InputSlot()

    Output = OutputSlot()
    PredictionProbabilities = OutputSlot()

    def __init__(self, *args, **kwargs):
        req_image = self.InputImage[:]
        req_image.submit()
        self.image = req_image.wait()
        
        req_labels = self.Labels[:]
        req_labels.submit()
        self.labels = req_labels.wait()
        
        self.opTrain = OpGraphCutsGMMTrain(image=self.image, labels=self.labels)
        self.opPredict = OpPredictGraphCutsGMM()

        self.opTrain.Image.connect( self.Image )
        self.opTrain.Labels.connect( self.Labels )
        self.opPredict.Classifier.connect( self.opTrain.Classifier )
        self.PredictionProbabilities.connect( self.opPredict.PMaps )

    def setupOutputs(self):
        pass
    
    def _simple_vh(self):
        unaries = np.array(self.PredictionProbabilities, dtype=np.int32)
        pairwise = -2 * np.eye(len(self.labels), dtype=np.int32)
        vertical = np.ones((unaries.shape[0], unaries.shape[1]), dtype=np.int32)
        horizontal = np.ones((unaries.shape[0], unaries.shape[1]), dtype=np.int32)
        return pygco.cut_simple_vh(unaries, pairwise, vertical, horizontal)

    def execute(self, slot, subindex, roi, result):
        # key = roi.toSlice()
        # raw = self.InputImage[key].wait()
        
        if slot.name == 'PredictionProbabilities':
            result[...] = self.PredictionProbabilities
        if slot.name == 'Output':
            result[...] = self._simple_vh()

    def propagateDirty(self, slot, subindex, roi):
        if slot.name == "InputImage":
            self.PredictionProbabilities.setDirty(roi)
            self.Output.setDirty(roi)
        else:
            assert False, "Unknown dirty input slot"

class OpGraphCutsGMMTrain(Operator):
    """
    Trains a Gaussian Mixture Model on all labeled pixels in an image.
    """

    name = "TrainGMMPixels"
    description = "Trains a Gaussian Mixture Model on all labeled pixels in an image."
    category = "Learning"

    Image = InputSlot()
    Labels = InputSlot()
    NonzeroLabelBlocks = InputSlot()
    Classifier = OutputSlot()

    def __init__(self, parent=None, *args, **kwargs):
        self._n_components = 3
        self._n_iter = 10
        self.image = kwargs.pop('image', None)
        self.labels = kwargs.pop('labels', None)
        assert len(kwargs) == 0, "OpGraphCutsGMMTrain received an unknown parameter"

    def execute(self, slot, subindex, roi, result):
        featList = []
        labelsList = []

        for i, labels in enumerate(self.Labels):
            if labels.meta.shape is not None:
                blocks = self.NonzeroLabelBlocks[i][0].wait()
                for b in blocks[0]:
                    request = labels[b]
                    featurekey = list(b)
                    featurekey[-1] = slice(None, None, None)
                    request2 = self.Image[i][featurekey]
                    reqlistlabels.append(request)
                    reqlistfeat.append(request2)

        for ir, req in enumerate(reqlistlabels):
            traceLogger.debug("Waiting for a label block...")
            labblock = req.wait()

            traceLogger.debug("Waiting for an image block...")
            image = reqlistfeat[ir].wait()

            indexes=numpy.nonzero(labblock[...,0].view(numpy.ndarray))
            features=image[indexes]
            labbla=labblock[indexes]

            featList.append(features)
            labelsList.append(labbla)         
        traceLogger.debug("Requests processed")

        if len(featList) == 0 or len(labelsList) == 0:
            # If there was no actual data to train with, we return None
            result[:] = None
        else:
            for i in range(len(self.Labels)):
                gmm = sklearn.mixture.GMM(n_components=self._n_components, covariance_type='full', n_iter=self._n_iter, init_params='wc')
                training_data = []
                result_simple_vh = simple_vh(img_file, labeled_pixels, labels)
                gmm.fit(featList)
                result.add(gmm)

        return result

    def propagateDirty(self, slot, subindex, roi):
        pass

class OpPredictGraphCutsGMM(Operator):
    name = "PredictGraphCutsGMM"
    description = "Predict on an image"
    category = "Learning"

    Image = InputSlot()
    Classifier = InputSlot()
    LabelsCount = InputSlot(stype='integer')
    PMaps = OutputSlot()

    """
    Taken from OpPredictRandomForest, not sure if applicable here
    """
    def setupOutputs(self):
        nlabels=self.inputs["LabelsCount"].value
        self.PMaps.meta.dtype = numpy.float32
        self.PMaps.meta.axistags = copy.copy(self.Image.meta.axistags)
        self.PMaps.meta.shape = self.Image.meta.shape[:-1]+(nlabels,) # FIXME: This assumes that channel is the last axis
        self.PMaps.meta.drange = (0.0, 1.0)

    def execute(self, slot, subindex, roi, result):
        key = roi.toSlice()
        nlabels = self.LabelsCount.value

        GMMs = self.Classifier[:].wait()

        if GMMs is None:
            # Training operator may return 'None' if there was no data to train with
            return numpy.zeros(numpy.subtract(roi.stop, roi.start), dtype=numpy.float32)[...]
        traceLogger.debug("OpPredictGraphCutsGMM: Got classifier")

        newKey = key[:-1]
        newKey += (slice(0,self.inputs["Image"].meta.shape[-1],None),)

        res = self.inputs["Image"][newKey].wait()

        shape = res.shape
        prod = numpy.prod(shape[:-1])
        res.shape = (prod, shape[-1])
        features = res

        predictions = [0]*len(GMMs)

        for i, gmm in enumerate(GMMs):
            predictions[i] = GMMs[i].score(numpy.asarray(features))

        result[:] = predictions
        return result

    def propagateDirty(self, slot, subindex, roi):
        key = roi.toSlice()
        if slot.name == "Classifier":
            logger.debug("OpPredictGraphCutsGMM: Classifier changed, setting dirty")
            if self.LabelsCount.ready() and self.LabelsCount.value > 0:
                self.PMaps.setDirty(slice(None,None,None))
        elif slot.name == "Image":
            nlabels = self.LabelsCount.value
            if nlabels > 0:
                self.PMaps.setDirty(key[:-1] + (slice(0,nlabels,None),))
        elif slot.name == "LabelsCount":
            # When the labels count changes, we must resize the output
            if self.configured():
                # FIXME: It's ugly that we call the 'private' _setupOutputs() function here,
                #  but the output shape needs to change when this input becomes dirty,
                #  and the output change needs to be propagated to the rest of the graph.
                self._setupOutputs()
            self.PMaps.setDirty(slice(None,None,None))