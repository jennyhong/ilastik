from lazyflow.graph import Operator, InputSlot, OutputSlot

# Python
import logging
logger = logging.getLogger(__name__)
traceLogger = logging.getLogger("TRACE." + __name__)

# additional modules to install
import sklearn.mixture
import numpy

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
        
        self.opTrain = OpObjectTrain(image=self.image, labels=self.labels)

    def setupOutputs(self):
        pass
    
    def execute(self, slot, subindex, roi, result):
        key = roi.toSlice()
        raw = self.InputImage[key].wait()
        
        if slot.name == 'PredictionProbabilities':
            result[...] = raw
        if slot.name == 'Output':
            result[...] = raw

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