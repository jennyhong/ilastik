from ilastik.workflow import Workflow

from lazyflow.graph import Graph

from ilastik.applets.dataSelection import DataSelectionApplet
from ilastik.applets.graphCuts import GraphCutsApplet

class GraphCutsWorkflow(Workflow):
    def __init__(self, headless, workflow_cmdline_args):
        # Create a graph to be shared by all operators
        graph = Graph()
        super(GraphCutsWorkflow, self).__init__(headless, graph=graph)
        self._applets = []

        # Create applets 
        self.dataSelectionApplet = DataSelectionApplet(self, "Input Data", "Input Data", supportIlastik05Import=True, batchDataGui=False)
        self.graphCutsApplet = GraphCutsApplet(self, "Graph Cuts Segmentation", "GraphCuts Segmentation")
        opDataSelection = self.dataSelectionApplet.topLevelOperator
        opDataSelection.DatasetRoles.setValue( ['Raw Data'] )

        self._applets.append( self.dataSelectionApplet )
        self._applets.append( self.graphCutsApplet )

    @property
    def applets(self):
        return self._applets

    @property
    def imageNameListSlot(self):
        return self.dataSelectionApplet.topLevelOperator.ImageName
