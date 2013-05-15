from lazyflow.graph import Graph

from ilastik.workflow import Workflow
from ilastik.applets.dataSelection import DataSelectionApplet
from ilastik.applets.tracking.manual.manualTrackingApplet import ManualTrackingApplet
from ilastik.applets.tracking.trainable.trainableTrackingApplet import TrainableTrackingApplet
from ilastik.applets.objectExtraction.objectExtractionApplet import ObjectExtractionApplet

class TrainableTrackingWorkflow( Workflow ):
    workflowName = "Tracking Workflow (trainable)"

    @property
    def applets(self):
        return self._applets
    
    @property
    def imageNameListSlot(self):
        return self.dataSelectionApplet.topLevelOperator.ImageName

    def __init__( self, *args, **kwargs ):
        graph = kwargs['graph'] if 'graph' in kwargs else Graph()
        if 'graph' in kwargs: del kwargs['graph']
        super(TrainableTrackingWorkflow, self).__init__(graph=graph, *args, **kwargs)
        
        ## Create applets 
        self.dataSelectionApplet = DataSelectionApplet(self, 
                                                       "Input Data", 
                                                       "Input Data", 
                                                       batchDataGui=False,
                                                       force5d=True)
        opSegDataSelection = self.dataSelectionApplet.topLevelOperator
        opSegDataSelection.DatasetRoles.setValue( ['Raw Data', 'Binary Segmentation'] )        
                                                                   
        self.objectExtractionApplet = ObjectExtractionApplet(workflow=self,
                                                                      name="Object Extraction")
        
        self.manualTrackingApplet = ManualTrackingApplet( workflow=self, name="Training" )
        self.learningApplet = TrainableTrackingApplet( workflow=self, name="Learning" )

        self._applets = []        
        self._applets.append(self.dataSelectionApplet)        
        self._applets.append(self.objectExtractionApplet)        
        self._applets.append(self.manualTrackingApplet)
        self._applets.append(self.learningApplet)
            
    def connectLane(self, laneIndex):
        opData = self.dataSelectionApplet.topLevelOperator.getLane(laneIndex)
        opObjExtraction = self.objectExtractionApplet.topLevelOperator.getLane(laneIndex)            
        opManualTracking = self.manualTrackingApplet.topLevelOperator.getLane(laneIndex)
        opLearning = self.learningApplet.topLevelOperator.getLane(laneIndex)
                
        ## Connect operators ##
        rawSlot = opData.ImageGroup[0]
        segSlot = opData.ImageGroup[1]
        opObjExtraction.RawImage.connect( rawSlot )
        opObjExtraction.BinaryImage.connect( segSlot )    
        
        opManualTracking.RawImage.connect( rawSlot )
        opManualTracking.LabelImage.connect( opObjExtraction.LabelImage )
        opManualTracking.BinaryImage.connect( segSlot )        
        opManualTracking.ObjectFeatures.connect( opObjExtraction.RegionFeatures )

        opLearning.Tracks.connect( opManualTracking.Labels )
        opLearning.Divisions.connect( opManualTracking.Divisions )
