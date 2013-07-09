from ilastik.applets.base.applet import Applet

from opBatchIo import OpBatchIo

from batchIoSerializer import BatchIoSerializer

from ilastik.utility import OpMultiLaneWrapper

class BatchIoApplet( Applet ):
    """
    This applet allows the user to select sets of input data, 
    which are provided as outputs in the corresponding top-level applet operator.
    """
    def __init__( self, workflow, title ):
        self._topLevelOperator = OpMultiLaneWrapper( OpBatchIo, parent=workflow,
                                     promotedSlotNames=set(['DatasetPath', 'ImageToExport', 'OutputFileNameBase', 'RawImage']) )
        super(BatchIoApplet, self).__init__(title, syncWithImageIndex=False)

        self._serializableItems = [ BatchIoSerializer(self._topLevelOperator, title) ]

        self._gui = None
        self._title = title
        
    @property
    def dataSerializers(self):
        return self._serializableItems

    @property
    def topLevelOperator(self):
        return self._topLevelOperator

    def getMultiLaneGui(self):
        if self._gui is None:
            from batchIoGui import BatchIoGui
            self._gui = BatchIoGui( self._topLevelOperator, self.guiControlSignal, self.progressSignal, self._title )
        return self._gui







