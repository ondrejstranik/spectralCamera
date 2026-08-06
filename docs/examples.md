# Examples

## Quickstart: fully virtual spectral microscope (`test_simpleSpectralMicroscope2`)

No hardware needed -- [`VirtualRGBCamera`](reference/instrument/sCamera/sCameraGenerator.md)
wires up a `VirtualCamera` + [`SCamera`](reference/instrument/sCamera/sCamera.md)
+ [`CalibrateRGBImage`](reference/algorithm/calibrateRGBImage.md) in one call,
and [`SimpleSpectralMicroscope`](reference/virtualSystem/simpleSpectralMicroscope.md)
synthesizes a believable raw frame for it every loop iteration:

```python
from spectralCamera.virtualSystem.simpleSpectralMicroscope import SimpleSpectralMicroscope
from spectralCamera.instrument.sCamera.sCameraGenerator import VirtualRGBCamera
from spectralCamera.gui.xywViewerGUI import XYWViewerGui
from viscope.gui.allDeviceGUI import AllDeviceGUI
from viscope.main import viscope

# camera + spectral processor + RGB calibration, wired together in one call
scs = VirtualRGBCamera(rgbOrder='RGB')
camera = scs.camera
sCamera = scs.sCamera

# forward model -- simulates the raw sensor frame for the virtual camera
vM = SimpleSpectralMicroscope()
vM.setVirtualDevice(camera)
vM.connect()

# GUI: raw-frame view + hyperspectral viewer
viewer = AllDeviceGUI(viscope)
viewer.setDevice(camera)
newGUI = XYWViewerGui(viscope)
newGUI.setDevice(sCamera)

viscope.run()
```

## Advanced: two cameras, a stage, and an integral-field calibration (`test_multiSpectralMicroscope`)

[`MultiSpectralMicroscope`](reference/virtualSystem/multiSpectralMicroscope.md)
drives a spectral camera *and* a plain camera *and* a stage at once, and
picks its dispersion routine to match whichever `CalibrateXImage` class the
spectral camera holds -- here `VirtualIFCamera` selects the integral-field
path:

```python
import numpy as np
from viscope.instrument.virtual.virtualCamera import VirtualCamera
from viscope.instrument.virtual.virtualStage import VirtualStage
from spectralCamera.instrument.sCamera.sCameraGenerator import VirtualIFCamera
from spectralCamera.virtualSystem.multiSpectralMicroscope import MultiSpectralMicroscope

camera2 = VirtualCamera(name='BWCamera')
camera2.connect()
camera2.setParameter('threadingNow', True)

scs = VirtualIFCamera()
camera, sCamera = scs.camera, scs.sCamera

stage = VirtualStage('stage')
stage.connect()

vM = MultiSpectralMicroscope()
vM.setVirtualDevice(sCamera=sCamera, camera2=camera2, stage=stage)
vM.sample.setCalibrationImage()
vM.sample.setSpectralCell()
vM.sample.position = np.array([-150, -250])
vM.connect()
```

## Minimal real-camera pipeline, one shot (`test_sCameraStatic`)

The same `SCamera` interface, but with a real
[`WebCamera`](reference/instrument/camera/webCamera/webCamera.md) and no
virtual system at all -- one frame in, one calibrated cube out:

```python
from spectralCamera.instrument.camera.webCamera.webCamera import WebCamera
from spectralCamera.instrument.sCamera.sCamera import SCamera
from spectralCamera.algorithm.calibrateRGBImage import CalibrateRGBImage
from spectralCamera.gui.spectralViewer.xywViewer import XYWViewer

camera = WebCamera(name='WebCamera')
camera.connect()
sCal = CalibrateRGBImage()

sCamera = SCamera(name='spectralWebCamera')
sCamera.connect(camera=camera)
sCamera.setParameter('calibrationData', sCal)

im = sCamera.getLastSpectralImage()
wavelength = sCamera.getParameter('wavelength')

sViewer = XYWViewer(im, wavelength)
sViewer.run()
```

## What actually differs between virtual and real

Just like in [viscope's own examples](https://github.com/ondrejstranik/viscope/blob/main/docs/examples.md#what-actually-differs),
swapping `VirtualRGBCamera`/`VirtualIFCamera` for `RGBWebCamera` (or any
other real-hardware [`SCameraGenerator`](reference/instrument/sCamera/sCameraGenerator.md)
constructor) only changes what supplies the raw frame -- a real Controller
(`cv2`, a vendor SDK) instead of a `SimpleSpectralMicroscope`/
`MultiSpectralMicroscope` virtual system. The `SCamera`, the calibration
object, and every GUI/viewer downstream of it are unchanged.

## Building a calibration object directly

Calibration objects don't need an `SCamera` at all -- they can be built and
called standalone, which is the easiest way to sanity-check one
(`test_calibrateIFImage`):

```python
from spectralCamera.algorithm.calibrateIFImage import CalibrateIFImage

sCal = CalibrateIFImage(camera=camera)   # camera: any connected BaseCamera
cube = sCal.getSpectralImage(rawImage)
```
