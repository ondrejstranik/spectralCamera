''' script to time MilCamera.getLastImage() for different exposureTime/nFrame
settings. getLastImage() already logs a grabWait/transfer/numpy-add
breakdown at INFO level, this script additionally times the call from
outside and compares it to the theoretical floor (nFrame * exposureTime). '''

#%% import and parameter definition

import logging
import time
import numpy as np
from spectralCamera.instrument.camera.milCamera.milCamera import MilCamera

logging.basicConfig(level=logging.INFO)

exposureTime = 10  # ms
nFrameList = [1, 10, 50, 100]
nRepeat = 3  # repeats per nFrame, to see run-to-run variation

#%% connect camera

cam = MilCamera(name='MilCamera')
cam.connect()
cam.setParameter('exposureTime', exposureTime)

#%% check what the camera actually accepted vs what we requested
# getLastImage's measured ~62 ms/frame (vs the requested 10 ms exposureTime)
# suggests the camera isn't actually running at the requested rate - read
# the GenICam features back to see what it really applied.

import mil as MIL

readExposureTime = MIL.MdigInquireFeature(cam.MilDigitizer, MIL.M_FEATURE_VALUE, "ExposureTime", MIL.M_TYPE_MIL_INT)
readFrameRate = MIL.MdigInquireFeature(cam.MilDigitizer, MIL.M_FEATURE_VALUE, "AcquisitionFrameRate", MIL.M_TYPE_MIL_INT)
readFramePeriod = MIL.MdigInquireFeature(cam.MilDigitizer, MIL.M_FEATURE_VALUE, "AcquisitionFramePeriod", MIL.M_TYPE_MIL_INT)

print(f'requested exposureTime: {exposureTime} ms ({exposureTime*1000} us)')
print(f'camera-reported ExposureTime: {readExposureTime} us')
print(f'camera-reported AcquisitionFrameRate: {readFrameRate}')
print(f'camera-reported AcquisitionFramePeriod: {readFramePeriod}')
print(f'computed AcquisitionFrameRate we sent: {cam.AcquisitionFrameRate.value}')
print(f'computed AcquisitionFramePeriod we sent: {cam.AcquisitionFramePeriod.value}')

#%% time getLastImage for each nFrame

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

#%% disconnect

cam.disconnect()

# %%
