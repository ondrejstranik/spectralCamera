# Installation

spectralCamera builds on [viscope](https://github.com/ondrejstranik/viscope).
Since neither package is on PyPI, viscope is installed straight from its
GitHub repository -- but only when requested through the `all` extra (see
below). Without it, viscope is **not** installed.

## For users

Use this if you just want to use spectralCamera in your own project (no need
to edit spectralCamera's own source).

0. (optional) create and activate a dedicated conda environment
   `conda create --name spectralCamera python=3.9` then
   `conda activate spectralCamera`
1. install spectralCamera directly from its GitHub repository, using the
   `all` extra so viscope is installed along with it
   `python -m pip install "spectralCamera[all] @ git+https://github.com/ondrejstranik/spectralCamera.git"`
   -- without the `[all]` extra, viscope is **not** installed.

This installs the latest version of both packages from their `main` branch.

2. to upgrade later, force a fresh install of both packages. A plain
   `--upgrade` is not enough for git dependencies -- pip only checks that the
   URL still matches, not whether the remote has new commits -- so
   `--force-reinstall` is required too (`--no-deps` keeps it from also
   reinstalling every other dependency):
   `python -m pip install --upgrade --force-reinstall --no-deps "spectralCamera[all] @ git+https://github.com/ondrejstranik/spectralCamera.git" "viscope @ git+https://github.com/ondrejstranik/viscope.git"`

## For developers

Use this if you want to modify spectralCamera itself -- the package is
installed in editable mode, so changes to the source take effect immediately
without reinstalling.

0. clone the repository and move into it
   `git clone https://github.com/ondrejstranik/spectralCamera.git` then
   `cd spectralCamera`
1. create and activate a conda environment
   `conda create --name spectralCamera python=3.9` then
   `conda activate spectralCamera`
2. install spectralCamera in editable mode

   - If you're only developing spectralCamera itself (not viscope), use the
     `all` extra so viscope gets installed for you from GitHub:
     `python -m pip install -e ".[all]"`
   - If you're also a developer of viscope and already have it installed
     locally (e.g. in editable mode from its own repo), leave out the extra
     so pip doesn't touch it:
     `python -m pip install -e .`
     See [viscope's installation guide](https://ondrejstranik.github.io/viscope/installation/)
     for how to install it in editable mode, so that changes to it take
     effect immediately too, the same way they do for spectralCamera.

If you use Pylance in VS Code, add the following to `.vscode\settings.json`
so it can resolve the package while it's installed in editable mode:
```
    "python.languageServer": "Pylance",
    "python.analysis.extraPaths": [
        "path\to\the\package\folder"
    ],
```

## Optional: Matrox (MIL) frame grabber support

[`MilCamera`](reference/instrument/camera/milCamera/milCamera.md) needs the
Matrox Imaging Library's Python wrapper, which is not on PyPI. If you have
MIL installed, add it into the same conda environment:
```
python -m pip install "C:\Program Files\Matrox Imaging\MIL\Scripting\pythonwrapper\dist\mil-10.50.923-py3-none-win_amd64.whl"
```
This is only needed if you're driving a camera through a MIL frame grabber
-- every other camera backend (webcam, Photonfocus, virtual) works without it.
