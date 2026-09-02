''' script to time MilCamera.getLastImage() for different exposureTime/nFrame/
roiHeight/roiOffsetY settings. getLastImage() already logs an on-board vs.
host breakdown at INFO level, this script additionally times the call from
outside and compares it to the theoretical floor (nFrame * exposureTime).

roiHeight/roiOffsetY must be set before connect() (they change the buffer/
image size), so each combination gets its own connect/disconnect cycle.
MilCamera resets Height/OffsetY/Binning/AcquisitionFrameRate/
AcquisitionFramePeriod to a known-clean state at the start of every
connect()/setParameter('exposureTime'), so results don't depend on whatever
a previous session (or CamExpert) left the camera at - all of these were
found to persist across connect()/disconnect() cycles, not just Height/
OffsetY.

Two things tried and confirmed NOT to speed up acquisition on this camera,
kept here as a record so they aren't retried blindly:
- BinningVertical: changed the image size/values, but not the readout time.
- DecimationVertical: the camera silently ignores it (readback stays 0).
Only cropping rows (roiHeight) actually raises the frame-rate ceiling here.

Also confirmed NOT to be safe to drop: leaving AcquisitionFrameRate/
AcquisitionFramePeriod unset ("the camera will just auto-maximize") - a
same-session A/B test made this look harmless once, because the untouched
run happened to coast on a fast value the previous run in the same script
execution had just set. Testing with a stale/leftover value from an
earlier, unrelated session exposed the real behavior: the camera can be
capped at whatever was left over (measured as slow as 4 fps at roiHeight
=1000, when it should be ~100 fps), so both must always be explicitly
(re)requested, matching the current exposure/ROI, every time. '''

#%% import and parameter definition

import logging
import time
import numpy as np
import mil as MIL
from spectralCamera.instrument.camera.milCamera.milCamera import MilCamera

logging.basicConfig(level=logging.INFO)

exposureTime = 10  # ms
nFrameList = [1, 10, 50, 100]
nRepeat = 3  # repeats per nFrame, to see run-to-run variation

# (roiHeight, roiOffsetY) combinations to sweep.
# roiHeight=None -> full sensor height; roiOffsetY=None -> centered crop.
settingsList = [
    (None, None),  # full sensor height
    (1000, None),  # 1000-row crop, centered
    (1000, 0),     # same crop size, but pinned to the top of the sensor
]

#%% define one connect -> measure -> disconnect sweep

def runSweep(roiHeight, roiOffsetY):
    cam = MilCamera(name='MilCamera')
    cam.roiHeight = roiHeight
    cam.roiOffsetY = roiOffsetY
    cam.connect()
    cam.setParameter('exposureTime', exposureTime)

    # what the camera actually accepted vs what we requested
    readExposureTime = MIL.MdigInquireFeature(cam.MilDigitizer, MIL.M_FEATURE_VALUE, "ExposureTime", MIL.M_TYPE_MIL_INT)
    readFrameRate = MIL.MdigInquireFeature(cam.MilDigitizer, MIL.M_FEATURE_VALUE, "AcquisitionFrameRate", MIL.M_TYPE_MIL_INT)
    readFramePeriod = MIL.MdigInquireFeature(cam.MilDigitizer, MIL.M_FEATURE_VALUE, "AcquisitionFramePeriod", MIL.M_TYPE_MIL_INT)
    readHeight = MIL.MdigInquireFeature(cam.MilDigitizer, MIL.M_FEATURE_VALUE, "Height", MIL.M_TYPE_MIL_INT)
    readOffsetY = MIL.MdigInquireFeature(cam.MilDigitizer, MIL.M_FEATURE_VALUE, "OffsetY", MIL.M_TYPE_MIL_INT)
    readBinningVertical = MIL.MdigInquireFeature(cam.MilDigitizer, MIL.M_FEATURE_VALUE, "BinningVertical", MIL.M_TYPE_MIL_INT)
    readBinningHorizontal = MIL.MdigInquireFeature(cam.MilDigitizer, MIL.M_FEATURE_VALUE, "BinningHorizontal", MIL.M_TYPE_MIL_INT)

    print(f'\n=== roiHeight requested={roiHeight}, roiOffsetY requested={roiOffsetY}, camera reports '
          f'Height={readHeight} OffsetY={readOffsetY} '
          f'BinningVertical={readBinningVertical} BinningHorizontal={readBinningHorizontal} '
          f'-> image size {cam.width} x {cam.height} ===')
    print(f'requested exposureTime: {exposureTime} ms ({exposureTime*1000} us)')
    print(f'camera-reported ExposureTime: {readExposureTime} us')
    print(f'camera-reported AcquisitionFrameRate: {readFrameRate}')
    print(f'camera-reported AcquisitionFramePeriod: {readFramePeriod}')

    for nFrame in nFrameList:
        cam.setParameter('nFrame', nFrame)

        floor_ms = nFrame * exposureTime
        print(f'\n--- nFrame={nFrame}, exposureTime={exposureTime} ms, floor={floor_ms:.1f} ms ---')

        times_ms = []
        for iRepeat in range(nRepeat):
            t0 = time.perf_counter()
            image = cam.getLastImage()
            dt_ms = (time.perf_counter() - t0) * 1000
            times_ms.append(dt_ms)
            print(f'  run {iRepeat}: {dt_ms:.1f} ms  (mean pixel value {np.mean(image):.1f})')

        print(f'  mean {np.mean(times_ms):.1f} ms, overhead over floor {np.mean(times_ms) - floor_ms:.1f} ms')

    cam.disconnect()

#%% run the sweep for each (roiHeight, roiOffsetY) combination

for roiHeight, roiOffsetY in settingsList:
    runSweep(roiHeight, roiOffsetY)

# %%
