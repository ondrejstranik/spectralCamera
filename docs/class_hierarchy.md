# Class Hierarchy

spectralCamera adds one standalone hierarchy of its own (calibration
algorithms) and several branches grafted onto viscope's existing
Instrument / Virtual System / GUI trees.

---

## Calibration Algorithm Layer (`spectralCamera/algorithm/`)

```
BaseCalibrate
├── CalibrateRGBImage        (Bayer RGGB / side-by-side RGB)
├── CalibrateFilterImage     (order x order spectral filter mosaic)
├── CalibrateIFImage         (integral-field / slanted-grating disperser)
├── CalibratePFImage         (Photonfocus CMV2K-SSM, wraps Photonfocus SDK)
├── CalibrateRamanImage      (Raman spectral-line grid, uses GridSuperPixel)
└── CalibrateFrom3Images     (3 narrow-band reference images, uses GridSuperPixel)

GridSuperPixel      (standalone -- shared grid/spot engine for the two above)
SpotSpectraSimple   (standalone -- per-spot spectra extraction, used by SViewer)
CalibrateLoader     (standalone -- classmethod load() for pickled calibration objects)
```

`BaseCalibrate` defines `getSpectralImage(rawImage) -> (wavelength, y, x)
cube` and `getWavelength()`; every subclass implements the sensor-specific
half of that mapping. `CalibrateRamanImage` and `CalibrateFrom3Images` don't
inherit from `GridSuperPixel` -- they each hold one and delegate the
grid-indexing/block-extraction work to it after fitting their own
pixel<->wavelength curve.

---

## Instrument Layer (`spectralCamera/instrument/`)

Grafted onto viscope's `BaseInstrument` tree
(`viscope.instrument.base.baseInstrument.BaseInstrument`):

```
BaseInstrument                         (viscope)
├── BaseProcessor                      (viscope)
│   └── SCamera
├── BaseSequencer                      (viscope)
│   └── SCameraFromFile
└── BaseCamera                         (viscope)
    ├── WebCamera
    ├── PFCamera
    └── MilCamera
```

`SCamera` and `SCameraFromFile` are not camera implementations themselves --
they consume a `BaseCamera`-family device and a `BaseCalibrate` object and
produce the hyperspectral cube. `WebCamera`, `PFCamera` and `MilCamera` are
ordinary `BaseCamera` implementations, each wrapping a different hardware
Controller (see [Software Architecture](architecture.md#instrument-layer-model)).

`HamamatsuCamera` and `baslerCamera` are not part of this tree -- they
predate the viscope port and still depend on the legacy `HSIplasmon`
package instead of `BaseCamera`.

`SCameraGenerator` is not a class hierarchy at all, just four factory
classes (`RGBWebCamera`, `VirtualRGBCamera`, `VirtualFilterCamera`,
`VirtualIFCamera`) that each construct one matching camera + `SCamera` +
calibration triple.

---

## Virtual System Layer (`spectralCamera/virtualSystem/`)

Grafted onto viscope's `BaseSystem`/`Component`/`Sample`:

```
BaseSystem                             (viscope)
├── SimpleSpectralMicroscope
└── MultiSpectralMicroscope

Component                              (viscope)
└── Component2

Sample                                 (viscope)
└── Sample2
```

`Component2` and `Sample2` extend rather than replace their viscope parents
-- `Component2` adds the dispersion routines specific to spectral sensor
geometries, `Sample2` adds the wavelength axis and synthetic spectral test
samples.

---

## GUI Layer (`spectralCamera/gui/`)

Grafted onto viscope's `BaseGUI` (a `QObject`):

```
BaseGUI                                (viscope)
├── SCameraGUI
├── SCameraFromFileGUI
├── SaveSIVideoGUI
├── SViewerGUI
└── XYWViewerGUI

QObject
├── SViewer          (wrapped by SViewerGUI)
└── XYWViewer        (wrapped by XYWViewerGUI)
```

`SViewer`/`XYWViewer` are plain `QObject`s, the same pattern viscope uses for
`NapariViewer`/`NapariGUI` -- the viewer holds the napari/pyqtgraph widgets,
the `*GUI` class is the thin adapter that a `ViewerWindow` can dock.
