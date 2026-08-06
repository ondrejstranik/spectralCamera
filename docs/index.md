# spectralCamera

[![GitHub](https://img.shields.io/badge/GitHub-spectralCamera-181717?logo=github)](https://github.com/ondrejstranik/spectralCamera)

spectralCamera is an extension package for
[viscope](https://github.com/ondrejstranik/viscope), the instrument-control
framework. Where viscope provides the generic device/GUI/virtual-microscope
machinery, spectralCamera adds everything specific to **spectral imaging**:
drivers for real spectral-camera hardware, the calibration algorithms that
turn a raw 2D sensor mosaic into a `(wavelength, y, x)` hyperspectral cube,
napari-based spectral viewers, and virtual systems that simulate a spectral
camera's raw output with no hardware attached.

A single raw frame can come from very different sensor geometries -- a
Bayer-filtered color chip, a spectral-filter mosaic, an integral-field
disperser, a Photonfocus filter-array sensor, or a Raman line-grid setup --
and spectralCamera's job is to hide that difference behind one interface:
[`SCamera`](reference/instrument/sCamera/sCamera.md) always exposes a
`(wavelength, y, x)` cube, regardless of which
[`CalibrateXImage`](architecture.md#calibration-algorithm-layer) class did
the conversion underneath.

See [Installation](installation.md) for setup instructions and [Software
Architecture](architecture.md) for how the pieces fit together on top of
viscope's Model/View/Controller split.
