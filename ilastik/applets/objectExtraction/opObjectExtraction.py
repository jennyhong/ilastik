#have to do that so that plugin manager has a handler
import ilastik.ilastik_logging
ilastik.ilastik_logging.default_config.init()

#Python
from copy import copy
import collections
from collections import defaultdict

#SciPy
import numpy as np
import vigra.analysis

#lazyflow
from lazyflow.graph import Operator, InputSlot, OutputSlot, OperatorWrapper
from lazyflow.stype import Opaque
from lazyflow.rtype import List
from lazyflow.roi import roiToSlice
from lazyflow.operators import OpCachedLabelImage, OpMultiArraySlicer2, OpMultiArrayStacker, OpArrayCache, OpCompressedCache

import logging
logger = logging.getLogger(__name__)

#ilastik
try:
    from ilastik.plugins import pluginManager
except:
    logger.warn('could not import pluginManager')

from ilastik.applets.base.applet import DatasetConstraintError

import collections

# These features are always calculated, but not used for prediction.
# They are needed by our gui, or by downstream applets.
default_features = ['Coord<Minimum>',
                    'Coord<Maximum>',
                    'RegionCenter',
                    'Count',
                ]

# to distinguish them, they go in their own category with this name
default_features_key = 'Default features'

def max_margin(d, default=(0, 0, 0)):
    """find any parameter named 'margin' in the nested feature
    dictionary 'd' and return the max.

    return 'default' if none are found.

    >>> max_margin({'plugin_one' : {'feature_one' : {'margin' : 10}}})
    [10, 10, 10]

    >>> max_margin({"p1": {"f1":{"margin":(10, 5, 2)}}})
    [10, 5, 2]

    """
    margin = default
    for features in d.itervalues():
        for params in features.itervalues():
            try:
                pmargin = params['margin']
                if not isinstance(pmargin, collections.Iterable):
                    pmargin = len(default)*[pmargin]
                margin = [max(x) for x in zip(margin, pmargin)]
            except (ValueError, KeyError):
                continue
    return margin

def make_bboxes(binary_bbox, margin):
    """Return binary label arrays for an object with margin.

    Helper for feature plugins.

    Returns (the object + context, context only)

    """
    # object and context
    max_margin = np.max(margin).astype(np.float32)
    scaled_margin = (max_margin / margin)
    if len(margin) > 2:
        dt = vigra.filters.distanceTransform3D(np.asarray(binary_bbox, dtype=np.float32),
                                               background=True,
                                               pixel_pitch=np.asarray(scaled_margin).astype(np.float64))
    else:
        dt = vigra.filters.distanceTransform2D(np.asarray(binary_bbox.squeeze(), dtype=np.float32),
                                               pixel_pitch=np.asarray(scaled_margin).astype(np.float64))
        dt = dt.reshape(dt.shape + (1,))

    assert dt.ndim == 3
    passed = np.asarray(dt < max_margin).astype(np.bool)

    # context only
    context = (passed - binary_bbox).astype(np.bool)
    return passed, context


class OpRegionFeatures3d(Operator):
    """Produces region features for a 3d image.

    The image MUST have xyzc axes, and is permitted to have t axis of dim 1.

    Inputs:

    * RawVolume : the raw data on which to compute features

    * LabelVolume : a volume of connected components for each object
      in the raw data.

    * Features : a nested dictionary of features to compute.
      Features[plugin name][feature name][parameter name] = parameter value

    Outputs:

    * Output : a nested dictionary of features.
      Output[plugin name][feature name] = numpy.ndarray
      
    """
    RawVolume = InputSlot()
    LabelVolume = InputSlot()
    Features = InputSlot(rtype=List, stype=Opaque)

    Output = OutputSlot()

    def setupOutputs(self):
        if self.LabelVolume.meta.axistags != self.RawVolume.meta.axistags:
            raise Exception('raw and label axis tags do not match')

        taggedOutputShape = self.LabelVolume.meta.getTaggedShape()
        taggedRawShape = self.RawVolume.meta.getTaggedShape()

        if not np.all(list(taggedOutputShape.get(k, 0) == taggedRawShape.get(k, 0)
                           for k in "txyz")):
            raise Exception("shapes do not match. label volume shape: {}."
                            " raw data shape: {}".format(self.LabelVolume.meta.shape,
                                                         self.RawVolume.meta.shape))

        if taggedOutputShape.get('t', 1) != 1:
            raise Exception('this operator cannot handle multiple time slices')
        if set(taggedOutputShape.keys()) - set('t') != set('xyzc'):
            raise Exception("Input volumes must have xyzc axes.")

        # Remove the spatial dims (keep t if present)
        del taggedOutputShape['x']
        del taggedOutputShape['y']
        del taggedOutputShape['z']
        del taggedOutputShape['c']

        self.Output.meta.shape = tuple(taggedOutputShape.values())
        self.Output.meta.axistags = vigra.defaultAxistags("".join(taggedOutputShape.keys()))
        # The features for the entire block (in xyz) are provided for the requested tc coordinates.
        self.Output.meta.dtype = object

    def execute(self, slot, subindex, roi, result):
        assert len(roi.start) == len(roi.stop) == len(self.Output.meta.shape)
        assert slot == self.Output

        # Process ENTIRE volume
        rawVolume = self.RawVolume[:].wait()
        labelVolume = self.LabelVolume[:].wait()

        # Convert to 4D (preserve axis order)
        axes4d = self.RawVolume.meta.getTaggedShape().keys()
        axes4d = filter(lambda k: k in 'xyzc', axes4d)

        rawVolume = rawVolume.view(vigra.VigraArray)
        rawVolume.axistags = self.RawVolume.meta.axistags
        rawVolume4d = rawVolume.withAxes(*axes4d)

        labelVolume = labelVolume.view(vigra.VigraArray)
        labelVolume.axistags = self.LabelVolume.meta.axistags
        labelVolume4d = labelVolume.withAxes(*axes4d)

        assert np.prod(roi.stop - roi.start) == 1
        acc = self._extract(rawVolume4d, labelVolume4d)
        result[tuple(roi.start)] = acc
        return result

    def compute_extent(self, i, image, mincoords, maxcoords, axes, margin):
        """Make a slicing to extract object i from the image."""
        #find the bounding box (margin is always 'xyz' order)
        result = [None] * 3
        minx = max(mincoords[i][axes.x] - margin[axes.x], 0)
        miny = max(mincoords[i][axes.y] - margin[axes.y], 0)

        # Coord<Minimum> and Coord<Maximum> give us the [min,max]
        # coords of the object, but we want the bounding box: [min,max), so add 1
        maxx = min(maxcoords[i][axes.x] + 1 + margin[axes.x], image.shape[axes.x])
        maxy = min(maxcoords[i][axes.y] + 1 + margin[axes.y], image.shape[axes.y])

        result[axes.x] = slice(minx, maxx)
        result[axes.y] = slice(miny, maxy)

        try:
            minz = max(mincoords[i][axes.z] - margin[axes.z], 0)
            maxz = min(maxcoords[i][axes.z] + 1 + margin[axes.z], image.shape[axes.z])
        except:
            minz = 0
            maxz = 1

        result[axes.z] = slice(minz, maxz)

        return result

    def compute_rawbbox(self, image, extent, axes):
        """essentially returns image[extent], preserving all channels."""
        key = copy(extent)
        key.insert(axes.c, slice(None))
        return image[tuple(key)]

    def _extract(self, image, labels):
        if not (image.ndim == labels.ndim == 4):
            raise Exception("both images must be 4D. raw image shape: {}"
                            " label image shape: {}".format(image.shape, labels.shape))

        # FIXME: maybe simplify?
        class Axes(object):
            x = image.axistags.index('x')
            y = image.axistags.index('y')
            z = image.axistags.index('z')
            c = image.axistags.index('c')
        axes = Axes()

        image = np.asarray(image, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.uint32)

        slc3d = [slice(None)] * 4 # FIXME: do not hardcode
        slc3d[axes.c] = 0

        labels = labels[slc3d]

        
        logger.debug("Computing default features")

        #FIXME: clamp the global vigra features here
        extrafeats = vigra.analysis.extractRegionFeatures(image[slc3d], labels,
                                                        default_features,
                                                        ignoreLabel=0)
        logger.debug("computed default features")

        extrafeats = dict((k.replace(' ', ''), v)
                          for k, v in extrafeats.iteritems())

        mincoords = extrafeats["Coord<Minimum>"]
        maxcoords = extrafeats["Coord<Maximum>"]
        nobj = mincoords.shape[0]

        feature_names = self.Features([]).wait()

        # do global features
        global_features = {}
        for plugin_name, feature_list in feature_names.iteritems():
            plugin = pluginManager.getPluginByName(plugin_name, "ObjectFeatures")
            global_features[plugin_name] = plugin.plugin_object.compute_global(image, labels, feature_list, axes)

        logger.debug("computing global features")
        # local features: loop over all objects
        def dictextend(a, b):
            for key in b:
                a[key].append(b[key])
            return a

        local_features = defaultdict(lambda: defaultdict(list))
        margin = max_margin(feature_names)
        if np.any(margin) > 0:
            for i in range(1, nobj):
                logger.debug("processing object {}".format(i))
                extent = self.compute_extent(i, image, mincoords, maxcoords, axes, margin)
                rawbbox = self.compute_rawbbox(image, extent, axes)
                binary_bbox = np.where(labels[tuple(extent)] == i, 1, 0).astype(np.bool)
                for plugin_name, feature_list in feature_names.iteritems():
                    plugin = pluginManager.getPluginByName(plugin_name, "ObjectFeatures")
                    feats = plugin.plugin_object.compute_local(rawbbox, binary_bbox, feature_list, axes)
                    local_features[plugin_name] = dictextend(local_features[plugin_name], feats)

        logger.debug("computing done, removing failures")
        # remove local features that failed
        for pname, pfeats in local_features.iteritems():
            for key in pfeats.keys():
                value = pfeats[key]
                try:
                    pfeats[key] = np.vstack(list(v.reshape(1, -1) for v in value))
                except:
                    logger.warn('feature {} failed'.format(key))
                    del pfeats[key]

        # merge the global and local features
        logger.debug("removed failed, merging")
        all_features = {}
        plugin_names = set(global_features.keys()) | set(local_features.keys())
        for name in plugin_names:
            d1 = global_features.get(name, {})
            d2 = local_features.get(name, {})
            all_features[name] = dict(d1.items() + d2.items())

        # reshape all features
        for pfeats in all_features.itervalues():
            for key, value in pfeats.iteritems():
                if value.shape[0] != nobj - 1:
                    raise Exception('feature {} does not have enough rows')

                # because object classification operator expects nobj to
                # include background. we should change that assumption.
                value = np.vstack((np.zeros(value.shape[1]),
                                   value))

                value = value.astype(np.float32) #turn Nones into numpy.NaNs

                assert value.dtype == np.float32
                assert value.shape[0] == nobj
                assert value.ndim == 2

                pfeats[key] = value
        logger.debug("merged, returning")
        # add features needed by downstream applets. these should be
        # removed before classification.
        all_features[default_features_key] = extrafeats
        return all_features

    def propagateDirty(self, slot, subindex, roi):
        if slot is self.Features:
            self.Output.setDirty(slice(None))
        else:
            axes = self.RawVolume.meta.getTaggedShape().keys()
            dirtyStart = collections.OrderedDict(zip(axes, roi.start))
            dirtyStop = collections.OrderedDict(zip(axes, roi.stop))

            # Remove the spatial and channel dims (keep t, if present)
            del dirtyStart['x']
            del dirtyStart['y']
            del dirtyStart['z']
            del dirtyStart['c']

            del dirtyStop['x']
            del dirtyStop['y']
            del dirtyStop['z']
            del dirtyStop['c']

            self.Output.setDirty(dirtyStart.values(), dirtyStop.values())

class OpRegionFeatures(Operator):
    """Computes region features on a 5D volume."""
    RawImage = InputSlot()
    LabelImage = InputSlot()
    Features = InputSlot(rtype=List, stype=Opaque)
    Output = OutputSlot()

    # Schematic:
    #
    # RawImage ----> opRawTimeSlicer ----
    #                                    \
    # LabelImage --> opLabelTimeSlicer --> opRegionFeatures3dBlocks --> opTimeStacker -> Output

    def __init__(self, *args, **kwargs):
        super(OpRegionFeatures, self).__init__(*args, **kwargs)

        # Distribute the raw data
        self.opRawTimeSlicer = OpMultiArraySlicer2(parent=self)
        self.opRawTimeSlicer.AxisFlag.setValue('t')
        self.opRawTimeSlicer.Input.connect(self.RawImage)
        assert self.opRawTimeSlicer.Slices.level == 1

        # Distribute the labels
        self.opLabelTimeSlicer = OpMultiArraySlicer2(parent=self)
        self.opLabelTimeSlicer.AxisFlag.setValue('t')
        self.opLabelTimeSlicer.Input.connect(self.LabelImage)
        assert self.opLabelTimeSlicer.Slices.level == 1

        self.opRegionFeatures3dBlocks = OperatorWrapper(OpRegionFeatures3d, operator_args=[], parent=self)
        assert self.opRegionFeatures3dBlocks.RawVolume.level == 1
        assert self.opRegionFeatures3dBlocks.LabelVolume.level == 1
        self.opRegionFeatures3dBlocks.RawVolume.connect(self.opRawTimeSlicer.Slices)
        self.opRegionFeatures3dBlocks.LabelVolume.connect(self.opLabelTimeSlicer.Slices)
        self.opRegionFeatures3dBlocks.Features.connect(self.Features)
        assert self.opRegionFeatures3dBlocks.Output.level == 1

        self.opTimeStacker = OpMultiArrayStacker(parent=self)
        self.opTimeStacker.AxisFlag.setValue('t')
        assert self.opTimeStacker.Images.level == 1
        self.opTimeStacker.Images.connect(self.opRegionFeatures3dBlocks.Output)

        # Connect our outputs
        self.Output.connect(self.opTimeStacker.Output)

    def setupOutputs(self):
        pass

    def execute(self, slot, subindex, roi, destination):
        assert False, "Shouldn't get here."

    def propagateDirty(self, slot, subindex, roi):
        pass # Nothing to do...

class OpCachedRegionFeatures(Operator):
    """Caches the region features computed by OpRegionFeatures."""
    RawImage = InputSlot()
    LabelImage = InputSlot()
    CacheInput = InputSlot(optional=True)
    Features = InputSlot(rtype=List, stype=Opaque)

    Output = OutputSlot()
    CleanBlocks = OutputSlot()

    # Schematic:
    #
    # RawImage -----   blockshape=(t,)=(1,)
    #               \                        \
    # LabelImage ----> OpRegionFeatures ----> OpArrayCache --> Output
    #                                                     \
    #                                                      --> CleanBlocks

    def __init__(self, *args, **kwargs):
        super(OpCachedRegionFeatures, self).__init__(*args, **kwargs)

        # Hook up the labeler
        self._opRegionFeatures = OpRegionFeatures(parent=self)
        self._opRegionFeatures.RawImage.connect(self.RawImage)
        self._opRegionFeatures.LabelImage.connect(self.LabelImage)
        self._opRegionFeatures.Features.connect(self.Features)

        # Hook up the cache.
        self._opCache = OpArrayCache(parent=self)
        self._opCache.Input.connect(self._opRegionFeatures.Output)

        # Hook up our output slots
        self.Output.connect(self._opCache.Output)
        self.CleanBlocks.connect(self._opCache.CleanBlocks)

    def setupOutputs(self):
        assert self.LabelImage.meta.axistags == self.RawImage.meta.axistags

        taggedOutputShape = self.LabelImage.meta.getTaggedShape()
        taggedRawShape = self.RawImage.meta.getTaggedShape()

        if not np.all(list(taggedOutputShape.get(k, 0) == taggedRawShape.get(k, 0)
                           for k in "txyz")):
            raise Exception("shapes do not match. label volume shape: {}."
                            " raw data shape: {}".format(self.LabelVolume.meta.shape,
                                                         self.RawVolume.meta.shape))


        # Every value in the regionfeatures output is cached seperately as it's own "block"
        blockshape = (1,) * len(self._opRegionFeatures.Output.meta.shape)
        self._opCache.blockShape.setValue(blockshape)

    def setInSlot(self, slot, subindex, roi, value):
        assert slot == self.CacheInput
        slicing = roiToSlice(roi.start, roi.stop)
        self._opCache.Input[ slicing ] = value

    def execute(self, slot, subindex, roi, destination):
        assert False, "Shouldn't get here."

    def propagateDirty(self, slot, subindex, roi):
        pass # Nothing to do...

class OpAdaptTimeListRoi(Operator):
    """Adapts the t array output from OpRegionFeatures to an Output
    slot that is called with a 'List' rtype, where the roi is a list
    of time slices, and the output is a dictionary of (time,
    featuredict) pairs.

    """
    Input = InputSlot()
    Output = OutputSlot(stype=Opaque, rtype=List)

    def setupOutputs(self):
        # Number of time steps
        self.Output.meta.shape = (self.Input.meta.getTaggedShape()['t'],)
        self.Output.meta.dtype = object

    def execute(self, slot, subindex, roi, destination):
        assert slot == self.Output, "Unknown output slot"
        taggedShape = self.Input.meta.getTaggedShape()

        # Special case: An empty roi list means "request everything"
        if len(roi) == 0:
            roi = range(taggedShape['t'])

        taggedShape['t'] = 1
        timeIndex = taggedShape.keys().index('t')

        result = {}
        for t in roi:
            start = [0] * len(taggedShape)
            stop = taggedShape.values()
            start[timeIndex] = t
            stop[timeIndex] = t + 1

            #FIXME: why is it wrapped like this?
            val = self.Input(start, stop).wait()
            assert val.shape == (1,)
            result[t] = val[0]

        return result

    def propagateDirty(self, slot, subindex, roi):
        assert slot == self.Input
        timeIndex = self.Input.meta.axistags.index('t')
        self.Output.setDirty(List(self.Output, range(roi.start[timeIndex], roi.stop[timeIndex])))

class OpObjectCenterImage(Operator):
    """Produceds an image with a cross in the center of each connected
    component.

    """
    BinaryImage = InputSlot()
    RegionCenters = InputSlot(rtype=List, stype=Opaque)
    Output = OutputSlot()

    def setupOutputs(self):
        self.Output.meta.assignFrom(self.BinaryImage.meta)

    @staticmethod
    def __contained_in_subregion(roi, coords):
        b = True
        for i in range(len(coords)):
            b = b and (roi.start[i] <= coords[i] and coords[i] < roi.stop[i])
        return b

    @staticmethod
    def __make_key(roi, coords):
        key = [int(coords[i] - roi.start[i]) for i in range(len(roi.start))]
        return tuple(key)

    def execute(self, slot, subindex, roi, result):
        assert slot == self.Output, "Unknown output slot"
        result[:] = 0
        for t in range(roi.start[0], roi.stop[0]):
            obj_features = self.RegionCenters([t]).wait()
            for ch in range(roi.start[-1], roi.stop[-1]):
                centers = obj_features[t][default_features_key]['RegionCenter']
                if centers.size:
                    centers = centers[1:, :]
                for center in centers:
                    x, y, z = center[0:3]
                    c = (t, x, y, z, ch)
                    if self.__contained_in_subregion(roi, c):
                        result[self.__make_key(roi, c)] = 1

        return result

    def propagateDirty(self, slot, subindex, roi):
        if slot is self.RegionCenters:
            self.Output.setDirty(slice(None))


class OpObjectExtraction(Operator):
    """The top-level operator for the object extraction applet.

    Computes object features and object center images.

    """
    name = "Object Extraction"

    RawImage = InputSlot()
    BinaryImage = InputSlot()
    BackgroundLabels = InputSlot()

    # which features to compute.
    # nested dictionary with format:
    # dict[plugin_name][feature_name][parameter_name] = parameter_value
    Features = InputSlot(rtype=List, stype=Opaque, value={})

    LabelImage = OutputSlot()
    ObjectCenterImage = OutputSlot()

    # the computed features.
    # nested dictionary with format:
    # dict[plugin_name][feature_name] = feature_value
    RegionFeatures = OutputSlot(stype=Opaque, rtype=List)

    # pass through the 'Features' input slot
    ComputedFeatureNames = OutputSlot(rtype=List, stype=Opaque)

    BlockwiseRegionFeatures = OutputSlot() # For compatibility with tracking workflow, the RegionFeatures output
                                           # has rtype=List, indexed by t.
                                           # For other workflows, output has rtype=ArrayLike, indexed by (t)

    LabelInputHdf5 = InputSlot(optional=True)
    LabelOutputHdf5 = OutputSlot()
    CleanLabelBlocks = OutputSlot()

    RegionFeaturesCacheInput = InputSlot(optional=True)
    RegionFeaturesCleanBlocks = OutputSlot()

    # Schematic:
    #
    # BackgroundLabels              LabelImage
    #                 \            /
    # BinaryImage ---> opLabelImage ---> opRegFeats ---> opRegFeatsAdaptOutput ---> RegionFeatures
    #                                   /                                     \
    # RawImage--------------------------                      BinaryImage ---> opObjectCenterImage --> opCenterCache --> ObjectCenterImage

    def __init__(self, *args, **kwargs):

        super(OpObjectExtraction, self).__init__(*args, **kwargs)

        # internal operators
        self._opLabelImage = OpCachedLabelImage(parent=self)
        self._opRegFeats = OpCachedRegionFeatures(parent=self)
        self._opRegFeatsAdaptOutput = OpAdaptTimeListRoi(parent=self)
        self._opObjectCenterImage = OpObjectCenterImage(parent=self)

        # connect internal operators
        self._opLabelImage.Input.connect(self.BinaryImage)
        self._opLabelImage.InputHdf5.connect(self.LabelInputHdf5)
        self._opLabelImage.BackgroundLabels.connect(self.BackgroundLabels)

        self._opRegFeats.RawImage.connect(self.RawImage)
        self._opRegFeats.LabelImage.connect(self._opLabelImage.Output)
        self._opRegFeats.Features.connect(self.Features)
        self.RegionFeaturesCleanBlocks.connect(self._opRegFeats.CleanBlocks)

        self._opRegFeats.CacheInput.connect(self.RegionFeaturesCacheInput)

        self._opRegFeatsAdaptOutput.Input.connect(self._opRegFeats.Output)

        self._opObjectCenterImage.BinaryImage.connect(self.BinaryImage)
        self._opObjectCenterImage.RegionCenters.connect(self._opRegFeatsAdaptOutput.Output)

        self._opCenterCache = OpCompressedCache(parent=self)
        self._opCenterCache.Input.connect(self._opObjectCenterImage.Output)

        # connect outputs
        self.LabelImage.connect(self._opLabelImage.Output)
        self.ObjectCenterImage.connect(self._opCenterCache.Output)
        self.RegionFeatures.connect(self._opRegFeatsAdaptOutput.Output)
        self.BlockwiseRegionFeatures.connect(self._opRegFeats.Output)
        self.LabelOutputHdf5.connect(self._opLabelImage.OutputHdf5)
        self.CleanLabelBlocks.connect(self._opLabelImage.CleanBlocks)
        self.ComputedFeatureNames.connect(self.Features)

        # As soon as input data is available, check its constraints
        self.RawImage.notifyReady( self._checkConstraints )
        self.BinaryImage.notifyReady( self._checkConstraints )

    def _checkConstraints(self, *args):
        if self.RawImage.ready() and self.BinaryImage.ready():
            if self.RawImage.meta.shape != self.BinaryImage.meta.shape:
                raise DatasetConstraintError( 
                    "Object Extraction",
                    "Raw Image shape {} does not match Binary Image shape {}.".format(
                        self.RawImage.meta.shape, self.BinaryImage.meta.shape ))

    def setupOutputs(self):
        taggedShape = self.RawImage.meta.getTaggedShape()
        for k in taggedShape.keys():
            if k == 't' or k == 'c':
                taggedShape[k] = 1
            else:
                taggedShape[k] = 256
        self._opCenterCache.BlockShape.setValue(tuple(taggedShape.values()))

    def execute(self, slot, subindex, roi, result):
        assert False, "Shouldn't get here."

    def propagateDirty(self, inputSlot, subindex, roi):
        pass

    def setInSlot(self, slot, subindex, roi, value):
        assert slot == self.LabelInputHdf5 or slot == self.RegionFeaturesCacheInput, "Invalid slot for setInSlot(): {}".format(slot.name)
        # Nothing to do here.
        # Our Input slots are directly fed into the cache,
        #  so all calls to __setitem__ are forwarded automatically
