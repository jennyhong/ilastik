from ilastik.applets.base.appletSerializer import \
    AppletSerializer, deleteIfPresent, SerialSlot, SerialClassifierSlot, \
    SerialBlockSlot, SerialListSlot
from lazyflow.operators.ioOperators import OpStreamingHdf5Reader, OpH5WriterBigDataset
import threading
from ilastik.utility.simpleSignal import SimpleSignal

import logging
logger = logging.getLogger(__name__)

class GraphCutsSerializer(AppletSerializer):
    
    def __init__(self, operator, projectFileGroupName):
        slots = [SerialSlot(operator.MinValue, selfdepends=True),
                 SerialSlot(operator.MaxValue, selfdepends=True)]
        
        super(GraphCutsSerializer, self).__init__(projectFileGroupName,
                                                         slots=slots)
