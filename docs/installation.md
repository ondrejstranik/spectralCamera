# Installation

spectralCamera builds on [viscope](https://github.com/ondrejstranik/viscope).
viscope is declared as a regular dependency (installed straight from its
GitHub repository, since neither package is on PyPI), so installing
spectralCamera pulls it in automatically -- there is no separate viscope
install step.

## For users

Use this if you just want to use spectralCamera in your own project (no need
to edit spectralCamera's own source).

0. (optional) create and activate a dedicated conda environment
   `conda create --name spectralCamera python=3.9` then
   `conda activate spectralCamera`
1. install spectralCamera directly from its GitHub repository
   `python -m pip install git+https://github.com/ondrejstranik/spectralCamera.git`
   -- this also installs viscope from its own GitHub repository, per the
   `viscope @ git+...` entry in spectralCamera's `dependencies`.

This installs the latest version of both packages from their `main` branch.
To upgrade later, re-run the same command with `--upgrade`.

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
   `python -m pip install -e .`
   -- this installs viscope too (non-editable, straight from GitHub). If you
   also plan to modify viscope itself, clone it separately and install it in
   editable mode instead, *before* step 2 -- pip will then see it already
   satisfied and won't overwrite it with the non-editable GitHub version.

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
