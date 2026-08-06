# Software Architecture

spectralCamera does not introduce a new architecture -- it plugs into
viscope's [Model / View / Controller layering](https://github.com/ondrejstranik/viscope/blob/main/docs/architecture.md)
and adds one thing viscope doesn't have an opinion about: turning a raw
sensor frame into a calibrated `(wavelength, y, x)` hyperspectral cube.
Everything in this package is either a Model class that fits into one of
viscope's existing layers, or a small **calibration algorithm layer** that
sits underneath the Model and has no viscope equivalent.

## Calibration algorithm layer

This layer has no viscope equivalent -- it sits underneath the Model, not
inside it. [`BaseCalibrate`](reference/algorithm/baseCalibrate.md) is the interface
every calibration algorithm implements: given a `wavelength` array and a raw
2D frame, `getSpectralImage(rawImage)` returns a `(wavelength, y, x)` cube.
It also provides `saveClass()`/`loadClass()` pickle persistence, so a fitted
calibration (e.g. one derived from a physical grid-alignment measurement)
can be reused across sessions without repeating the fit.

Each subclass targets a different sensor geometry:

- [`CalibrateRGBImage`](reference/algorithm/calibrateRGBImage.md) -- a
  standard colour sensor, either a Bayer `RGGB` mosaic (averages the two
  green sub-pixels) or three side-by-side monochrome frames (`RGB` mode).
- [`CalibrateFilterImage`](reference/algorithm/calibrateFilterImage.md) --
  an `order x order` spectral filter mosaic (each block of `order**2`
  pixels is one wavelength-encoded super-pixel).
- [`CalibrateIFImage`](reference/algorithm/calibrateIFImage.md) -- an
  integral-field sensor (microlens array + slanted-grating disperser):
  purely geometric, walks a known lattice (`position00`, `gridVector`) to
  read each dispersed line as one spectrum.
- [`CalibratePFImage`](reference/algorithm/calibratePFImage.md) -- wraps the
  vendor `Photonfocus` driver for the CMV2K-SSM Fabry-Perot filter-mosaic
  sensor; the actual pixel-to-wavelength mapping lives in the vendor SDK,
  this class just adapts it to `BaseCalibrate`.
- [`CalibrateRamanImage`](reference/algorithm/calibrateRamanImage.md) /
  [`CalibrateFrom3Images`](reference/algorithm/calibrateFrom3Images.md) --
  for sensors that image a *grid of dispersed spectral blocks* rather than a
  regular mosaic (a Raman line-grating setup, or three narrow-band
  reference images). Both fit an arbitrarily rotated/skewed lattice and a
  pixel<->wavelength curve from calibration data, then reuse the shared grid
  engine below to extract spectra at runtime.

[`GridSuperPixel`](reference/algorithm/gridSuperPixel.md) is that shared grid
engine: given raw spot positions it finds the lattice basis vectors (via
[`basisVectors.lattice_basis_vectors`](reference/algorithm/basisVectors.md)),
assigns each spot a `(row, col)` grid index, and extracts/averages the pixel
block around each spot into one cube. `CalibrateRamanImage` and
`CalibrateFrom3Images` are both thin wrappers around this engine that differ
only in *how* they fit the wavelength axis.

[`CalibrateLoader`](reference/algorithm/calibrateLoader.md) unpickles a saved
calibration object by path, falling back to a default `CalibrateRGBImage` if
the path doesn't exist. It currently imports its calibration classes from
the legacy `HSIplasmon` package rather than from `spectralCamera` itself --
treat it, along with [`HamamatsuCamera`](reference/instrument/HamamatsuCamera.md)
and [`baslerCamera`](reference/instrument/baslerCamera.md) (same `HSIplasmon`
dependency, and not viscope `BaseCamera` subclasses), as not-yet-ported
legacy code rather than a supported code path.

## Instrument layer (Model)

[`SCamera`](reference/instrument/sCamera/sCamera.md) extends viscope's
`BaseProcessor`. It is the glue between a plain `BaseCamera` and a
calibration object: it holds a `camera` and a `spectraCalibration`
(any `BaseCalibrate`), and on every new raw frame calls
`imageDataToSpectralCube()` to produce `sImage`, the hyperspectral cube the
rest of the package consumes. It also owns spectral-video recording via
[`FileSIVideo`](reference/algorithm/fileSIVideo.md).

[`SCameraFromFile`](reference/instrument/sCamera/sCameraFromFile.md) extends
viscope's `BaseSequencer` and stands in for `SCamera` when developing without
a live camera: it replays a previously recorded sequence (`FileSIVideo` or,
for LCTF-tuned-filter acquisitions,
[`FileSILCTFVideo`](reference/algorithm/fileSILCTFVideo.md)) frame by frame.

[`SCameraGenerator`](reference/instrument/sCamera/sCameraGenerator.md)
provides four convenience constructors (`RGBWebCamera`, `VirtualRGBCamera`,
`VirtualFilterCamera`, `VirtualIFCamera`) that each wire up a camera (real or
virtual) together with the matching `SCamera` + calibration pair in one call
-- see [Examples](examples.md).

Real hardware wrappers extend viscope's `BaseCamera` directly, each calling
into its own Controller SDK:

| Class | Wraps | Controller |
|---|---|---|
| [`WebCamera`](reference/instrument/camera/webCamera/webCamera.md) | USB/integrated webcam | `cv2.VideoCapture` |
| [`PFCamera`](reference/instrument/camera/pfCamera/pFCamera.md) | Photonfocus CMV2K filter-mosaic camera | `Photonfocus` / PFPyCameraLib SDK |
| [`MilCamera`](reference/instrument/camera/milCamera/milCamera.md) | camera behind a Matrox (MIL) frame grabber | Matrox Imaging Library |

`SCamera`/`SCameraFromFile` are the Model classes a GUI talks to; the
hardware wrappers above are also Model classes (they still expose the
`BaseCamera` interface), but internally they call out to the Controller
libraries in the table -- exactly the same split as `WebCamera` in viscope's
own architecture doc.

## Virtual system layer (Model -- simulation, test/dev only)

[`SimpleSpectralMicroscope`](reference/virtualSystem/simpleSpectralMicroscope.md)
and [`MultiSpectralMicroscope`](reference/virtualSystem/multiSpectralMicroscope.md)
extend viscope's `BaseSystem` the same way `SimpleMicroscope` does: a
threaded loop that recomputes a virtual device's raw frame whenever its
parameters change. `SimpleSpectralMicroscope` drives one virtual camera;
`MultiSpectralMicroscope` drives a spectral camera, a second plain camera,
and optionally a stage, and dispatches to a different dispersion routine
depending on which `CalibrateXImage` class the spectral camera is using --
the simulator has to know the sensor geometry it needs to *produce*, mirroring
the calibration algorithm that will later decode it.

[`Component2`](reference/virtualSystem/component/component2.md) extends
viscope's `Component` with the spectral optics helpers those simulators need
(`disperseHorizontal`, `disperseIntoBlock`, `disperseIntoRGGBBlock`,
`disperseIntoLines`, `spectraRangeAdjustment`) -- one per sensor geometry in
the calibration algorithm layer, run in reverse.

[`Sample2`](reference/virtualSystem/component/sample2.md) extends viscope's
`Sample` with a wavelength axis and several ready-made synthetic samples
(`setSpectralAstronaut`, `setSpectralCell`, `setSpectralDisk`,
`setCalibrationImage`) so a virtual spectral microscope has something
plausible to image without any real sample or hardware.

## GUI layer (View)

All panels below extend viscope's `BaseGUI` and attach to a device the same
way viscope's own device GUIs do:

- [`SCameraGUI`](reference/gui/sCameraGUI.md) -- parameter panel for an
  `SCamera` (aberration correction, spectral smoothing, dark value).
- [`SCameraFromFileGUI`](reference/gui/sCameraFromFileGUI.md) -- folder
  picker and playback controls for an `SCameraFromFile`.
- [`SaveSIVideoGUI`](reference/gui/saveSIVideoGUI.md) -- start/stop
  recording a spectral video sequence to disk.
- [`SViewerGUI`](reference/gui/sViewerGUI.md) and
  [`XYWViewerGUI`](reference/gui/xywViewerGUI.md) -- viewer panels, each
  wrapping a standalone napari-based viewer
  ([`SViewer`](reference/gui/spectralViewer/sViewer.md) /
  [`XYWViewer`](reference/gui/spectralViewer/xywViewer.md)) that shows the
  hyperspectral cube plus a per-point spectra plot. The viewers themselves
  are plain `QObject`s, not `BaseGUI` -- the `*GUI` classes are the thin
  viscope-facing adapters, the same split viscope uses for `NapariViewer` /
  `NapariGUI`.

Because every piece above still speaks `BaseCamera`/`BaseGUI`, the same
distinction viscope's architecture doc makes for `VirtualCamera` vs.
`WebCamera` holds here one level up: a GUI or virtual-system test written
against `VirtualRGBCamera` runs unmodified against a real `RGBWebCamera` --
see [Examples](examples.md).
