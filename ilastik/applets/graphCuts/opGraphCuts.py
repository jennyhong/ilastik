from lazyflow.graph import Operator, InputSlot, OutputSlot

import numpy

# additional modules to install
import sklearn.mixture

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

    Classifier = OutputSlot()

    def __init__(self, *args, **kwargs):
        self.image = kwargs.pop('image', None)
        self.labels = kwargs.pop('labels', None)
        assert len(kwargs) == 0, "OpGraphCutsGMMTrain received an unknown parameter"

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