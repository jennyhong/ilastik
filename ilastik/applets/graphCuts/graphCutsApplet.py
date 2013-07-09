from ilastik.applets.base.standardApplet import StandardApplet

from opGraphCuts import OpGraphCuts
from graphCutsSerializer import GraphCutsSerializer

class GraphCutsApplet( StandardApplet ):
    """
    Applet for graph cuts-based segmentation
    """
    def __init__( self, workflow, guiName, projectFileGroupName ):
        super(GraphCutsApplet, self).__init__(guiName, workflow)
        self._serializableItems = [ GraphCutsSerializer(self.topLevelOperator, projectFileGroupName) ]
        
    @property
    def singleLaneOperatorClass(self):
        return OpGraphCuts
    
    @property
    def singleLaneGuiClass(self):
        from graphCutsGui import GraphCutsGui
        return GraphCutsGui

    @property
    def dataSerializers(self):
        return self._serializableItems
