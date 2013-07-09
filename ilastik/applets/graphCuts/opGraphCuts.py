from lazyflow.graph import Operator, InputSlot, OutputSlot

import numpy

# additional modules to install
import sklearn.mixture

class OpGraphCuts(Operator):

    name = "OpGraphCuts"
    category = "Pointwise"

    InputImage = InputSlot()
    LabelInputs = InputSlot(optional = True)
    FeatureImages = InputSlot()

    PredictionProbabilities = OutputSlot()

    def __init__(self, *args, **kwargs):
        self.opTrain = OpObjectTrain(parent=self)

    def setupOutputs(self):
        pass
    
    def execute(self, slot, subindex, roi, result):
        key = roi.toSlice()
        raw = self.InputImage[key].wait()
        mask = numpy.logical_and(self.MinValue.value <= raw, raw <= self.MaxValue.value)
        
        if slot.name == 'Output':
            result[...] = mask * raw
        if slot.name == 'InvertedOutput':
            result[...] = numpy.logical_not(mask) * raw

    def propagateDirty(self, slot, subindex, roi):
        if slot.name == "InputImage":
            self.PredictionProbabilities.setDirty(roi)
        else:
            assert False, "Unknown dirty input slot"

class OpGraphCutsGMMTrain(Operator):
    """
    Trains a Gaussian Mixture Model on all labeled pixels in an image.
    """

    name = "TrainGMMPixels"
    description = "Trains a Gaussian Mixture Model on all labeled pixels in an image."
    category = "Learning"

    Labels = InputSlot()
    Features = InputSlot()

    Classifier = OutputSlot()

    def execute(self, slot, subindex, roi, result):
        featList = []
        all_col_names = []
        labelsList = []
        result = []

        for i in range(len(self.Labels)):
            feats = self.Features[i]([]).wait()
            gmm = sklearn.mixture.GMM(n_components=3, covariance_type='full', n_iter=10, init_params='wc')
            training_data = []
            training_data = get_training_data(feats, i)
            gmm.fit(training_data)
            result.add(gmm)

        return result

    def propagateDirty(self, slot, subindex, roi):
        pass