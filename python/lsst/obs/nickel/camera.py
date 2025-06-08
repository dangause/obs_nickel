# camera.py

from lsst.afw.cameraGeom import DetectorType, Camera, Detector
from lsst.afw.cameraGeom import AmplifierBuilder, DetectorBuilder
from lsst.afw.cameraGeom import Orientation
from lsst.geom import Box2I, Point2D, Extent2D

def makeCamera():
    # Define a single amplifier (minimal)
    ampBuilder = AmplifierBuilder()
    ampBuilder.setName("AMP1")
    ampBuilder.setBBox(Box2I((0, 0), (1055, 1023)))
    ampBuilder.setGain(1.0)
    ampBuilder.setReadNoise(5.0)
    ampBuilder.setSaturation(65535)
    ampBuilder.setHasRawInfo(False)
    amplifier = ampBuilder.finish()

    # Define detector builder
    detBuilder = DetectorBuilder("ccd1")
    detBuilder.setId(0)
    detBuilder.setSerial("0")
    detBuilder.setDetectorType(DetectorType.SCIENCE)
    detBuilder.setBBox(Box2I((0, 0), (1055, 1023)))
    detBuilder.setPixelSize(15.0, 15.0)
    detBuilder.setOrientation(Orientation())
    detBuilder.append(amplifier)

    # Build detector and camera
    detector = detBuilder.finish()
    camera = Camera("NickelCam", [detector])
    return camera
