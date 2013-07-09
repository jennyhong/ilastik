import os
import tempfile
import h5py
import numpy.random
from lazyflow.graph import Graph
from lazyflow.operators import OpGaussianSmoothing
from lazyflow.operators.ioOperators import OpInputDataReader
from ilastik.applets.batchIo.opBatchIo import OpBatchIo, ExportFormat
from ilastik.applets.dataSelection.opDataSelection import DatasetInfo
from ilastik.utility import PathComponents

class TestOpBatchIo(object):
    def setUp(self):
        tempDir = tempfile.gettempdir()
        os.chdir(tempDir)
        self.testDataFileName = 'NpyTestData.npy'
        
    def tearDown(self):
        try:
            os.remove(self.testDataFileName)
        except:
            pass
    
    def testBasic5d(self):
        # Start by writing some test data to disk.
        self.testData = numpy.random.random((1,10,10,10,2)).astype(numpy.float32)
        self.expectedDataShape = (1,10,10,10,2)
        numpy.save(self.testDataFileName, self.testData)
        
        self.basicImpl()

    def testBasic4d(self):
        # Start by writing some 4D test data to disk.
        self.testData = numpy.random.random((10,10,10,2)).astype(numpy.float32)
        self.expectedDataShape = (10,10,10,2)
        numpy.save(self.testDataFileName, self.testData)

        self.basicImpl()
        
    def testBasic3d(self):
        # Start by writing some 4D test data to disk.
        self.testData = numpy.random.random((10,10,10)).astype(numpy.float32)
        self.expectedDataShape = (10,10,10,1)
        numpy.save(self.testDataFileName, self.testData)

        self.basicImpl()
        
    def testBasic2d(self):
        # Start by writing some 4D test data to disk.
        self.testData = numpy.random.random((10,10)).astype(numpy.float32)
        numpy.save(self.testDataFileName, self.testData)
        self.expectedDataShape = (10,10,1)

        self.basicImpl()
        
    def basicImpl(self):
        cwd = os.getcwd()
        info = DatasetInfo()
        info.filePath = os.path.join(cwd, self.testDataFileName)
        
        graph = Graph()
        opBatchIo = OpBatchIo(graph=graph)
        opInput = OpInputDataReader(graph=graph)
        opInput.FilePath.setValue( info.filePath )
        
        # Our test "processing pipeline" is just a smoothing operator.
        opSmooth = OpGaussianSmoothing(graph=graph)
        opSmooth.Input.connect( opInput.Output )
        opSmooth.sigma.setValue(3.0)
        
        opBatchIo.ExportDirectory.setValue( '' )
        opBatchIo.Suffix.setValue( '_smoothed' )
        opBatchIo.Format.setValue( ExportFormat.H5 )
        opBatchIo.DatasetPath.setValue( info.filePath )
        opBatchIo.WorkingDirectory.setValue( cwd )
        
        internalPath = 'path/to/data'
        opBatchIo.InternalPath.setValue( internalPath )
        
        opBatchIo.ImageToExport.connect( opSmooth.Output )
        
        dirty = opBatchIo.Dirty.value
        assert dirty == True
        
        outputPath = opBatchIo.OutputDataPath.value
        assert outputPath == os.path.join(cwd, 'NpyTestData_smoothed.h5/' + internalPath)
        
        result = opBatchIo.ExportResult.value
        assert result
        
        dirty = opBatchIo.Dirty.value
        assert dirty == False
        
        # Check the file
        smoothedPath = os.path.join(cwd, 'NpyTestData_smoothed.h5')
        with h5py.File(smoothedPath, 'r') as f:
            assert internalPath in f
            assert f[internalPath].shape == self.expectedDataShape
            assert (f[internalPath][:] == opSmooth.Output[:].wait()).all()
        try:
            os.remove(smoothedPath)
        except:
            pass
        
        # Check the exported image
        assert ( opBatchIo.ExportedImage[:].wait() == opSmooth.Output[:].wait() ).all()

    def testCreateExportDirectory(self):
        """
        Test that the batch operator can create the export directory if it doesn't exist yet.
        """
        # Start by writing some test data to disk.
        self.testData = numpy.random.random((1,10,10,10,1))
        numpy.save(self.testDataFileName, self.testData)

        cwd = os.getcwd()
        info = DatasetInfo()
        info.filePath = os.path.join(cwd, 'NpyTestData.npy')
        
        graph = Graph()
        opBatchIo = OpBatchIo(graph=graph)
        opInput = OpInputDataReader(graph=graph)
        opInput.FilePath.setValue( info.filePath )
        
        # Our test "processing pipeline" is just a smoothing operator.
        opSmooth = OpGaussianSmoothing(graph=graph)
        opSmooth.Input.connect( opInput.Output )
        opSmooth.sigma.setValue(3.0)
        
        exportDir = os.path.join(cwd, 'exported_data')
        opBatchIo.ExportDirectory.setValue( exportDir )
        opBatchIo.Suffix.setValue( '_smoothed' )
        opBatchIo.Format.setValue( ExportFormat.H5 )
        opBatchIo.DatasetPath.setValue( info.filePath )
        opBatchIo.WorkingDirectory.setValue( cwd )
        
        internalPath = 'path/to/data'
        opBatchIo.InternalPath.setValue( internalPath )
        
        opBatchIo.ImageToExport.connect( opSmooth.Output )
        
        dirty = opBatchIo.Dirty.value
        assert dirty == True
        
        outputPath = opBatchIo.OutputDataPath.value
        assert outputPath == os.path.join(exportDir, 'NpyTestData_smoothed.h5', internalPath)
        
        result = opBatchIo.ExportResult.value
        assert result
        
        dirty = opBatchIo.Dirty.value
        assert dirty == False
        
        # Check the file
        smoothedPath = PathComponents(outputPath).externalPath
        with h5py.File(smoothedPath, 'r') as f:
            assert internalPath in f
            assert f[internalPath].shape == self.testData.shape
        try:
            os.remove(smoothedPath)
            os.rmdir(exportDir)
        except:
            pass
        
if __name__ == "__main__":
    import sys
    import nose
    sys.argv.append("--nocapture")    # Don't steal stdout.  Show it on the console as usual.
    sys.argv.append("--nologcapture") # Don't set the logging level to DEBUG.  Leave it alone.
    nose.run(defaultTest=__file__)
