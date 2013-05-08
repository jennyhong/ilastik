from lazyflow.graph import Graph

from ilastik.workflow import Workflow
from ilastik.workflows.tracking.manual.manualTrackingWorkflow import ManualTrackingWorkflow


class TrainableTrackingWorkflow( Workflow ):
    workflowName = "Tracking (trainable)"

    @property
    def applets( self ):
        return self._manual_wf.applets

    @property
    def imageNameListSlot( self ):
        return self._manual_wf.imageNameListSlot

    def __init__( self, headless, *args, **kwargs ):
        graph = kwargs['graph'] if 'graph' in kwargs else Graph()
        if 'graph' in kwargs: del kwargs['graph']
        super(TrainableTrackingWorkflow, self).__init__(headless=headless, graph=graph, *args, **kwargs)

        self._manual_wf = ManualTrackingWorkflow( headless, *args, **kwargs )

        

    def connectLane( self, laneIndex ):
        self._manual_wf.connectLane( laneIndex )


    


        
        


        
        
    
    
