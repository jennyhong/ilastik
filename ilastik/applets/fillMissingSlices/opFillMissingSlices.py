from lazyflow.graph import Operator, InputSlot, OutputSlot
from lazyflow.operators import OpInterpMissingData, OpBlockedArrayCache

import logging
loggerName = __name__ 
logger = logging.getLogger(loggerName)
logger.setLevel(logging.DEBUG)

class OpFillMissingSlicesNoCache(Operator):
    Input = InputSlot()
    DetectionMethod = InputSlot(value='classic')
    Output = OutputSlot()
    Missing = OutputSlot()
    
    def __init__(self, *args, **kwargs):
        super( OpFillMissingSlicesNoCache, self ).__init__(*args, **kwargs)
        
        # Set up interpolation
        self._opInterp = OpInterpMissingData( parent=self )
        self._opInterp.InputVolume.connect( self.Input )
        self._opInterp.InputSearchDepth.setValue(100)
        
        self._opInterp.DetectionMethod.connect(self.DetectionMethod)

        self.Output.connect( self._opInterp.Output )
        self.Missing.connect( self._opInterp.Missing )
        
    def execute(self, slot, subindex, roi, result):
        assert False, "Shouldn't get here"
    
    def propagateDirty(self, slot, subindex, roi):
        pass # Nothing to do here.
    
    def isDirty(self):
        #FIXME
        return True
    
    def dumps(self):
        return self._opInterp.dumps()
    
    def loads(self, s):
        self._opInterp.loads(s)
    
    
    

class OpFillMissingSlices(OpFillMissingSlicesNoCache):
    """
    Extends the cacheless operator above with a cached output.
    Suitable for use in a GUI, but not in a headless workflow.
    """
    CachedOutput = OutputSlot()

    def __init__(self, *args, **kwargs):
        super( OpFillMissingSlices, self ).__init__(*args, **kwargs)

        # The cache serves two purposes:
        # 1) Determine shape of accesses to the interpolation operator
        # 2) Avoid duplicating work
        self._opCache = OpBlockedArrayCache( parent=self )
        self._opCache.Input.connect( self._opInterp.Output )
        self._opCache.fixAtCurrent.setValue( False )

        self.CachedOutput.connect( self._opCache.Output )

    def setupOutputs(self):
        blockdims = { 't' : 1,
                      'x' : 256,
                      'y' : 256,
                      'z' : 100,
                      'c' : 1 }
        blockshape = map( blockdims.get, self.Input.meta.getTaggedShape().keys() )        
        self._opCache.innerBlockShape.setValue( tuple(blockshape) )        
        self._opCache.outerBlockShape.setValue( tuple(blockshape) )

