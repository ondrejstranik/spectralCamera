# -*- coding: utf-8 -*-
"""
Camera with the mil grabber card

Created on Fri Aug 25 08:44:55 2023

@author: ungersebastian / ondrej Stranik
"""
#%%

import logging
import time
import threading
import mil as MIL
import numpy as np
import ctypes
#from os import path, mkdir
#from timeit import default_timer as timer


from viscope.instrument.base.baseCamera import BaseCamera

logger = logging.getLogger(__name__)


class _AccumulateHookData:
    ''' state shared between MilCamera.getLastImage() and the MdigProcess
    hook callback that runs on MIL's internal grab thread. '''

    def __init__(self, nFrame, accBuffer, flushEvery, height, width):
        self.nFrame = nFrame
        self.accBuffer = accBuffer
        self.flushEvery = flushEvery
        self.count = 0
        self.unflushedCount = 0
        self.total = np.zeros((height, width), dtype=np.float64)
        self.done = threading.Event()

    def flush(self):
        ''' transfer the on-board partial sum to the host and reset it '''
        if self.unflushedCount == 0:
            return
        self.total += MIL.MbufGet(self.accBuffer)
        MIL.MbufClear(self.accBuffer, MIL.M_COLOR_BLACK)
        self.unflushedCount = 0


def _accumulateFrameHook(HookType, HookId, HookDataPtr):
    ''' called by MIL once per grabbed frame. Stays on-board (MimArith add)
    and only transfers to the host every flushEvery frames, so it can keep
    up with continuous grabbing instead of falling behind it. '''
    hookData = HookDataPtr
    if hookData.count >= hookData.nFrame:
        return 0

    bufferId = MIL.MdigGetHookInfo(HookId, MIL.M_MODIFIED_BUFFER + MIL.M_BUFFER_ID)
    MIL.MimArith(bufferId, hookData.accBuffer, hookData.accBuffer, MIL.M_ADD)
    hookData.count += 1
    hookData.unflushedCount += 1

    if hookData.unflushedCount >= hookData.flushEvery:
        hookData.flush()

    if hookData.count >= hookData.nFrame:
        hookData.done.set()

    return 0


class MilCamera(BaseCamera):
    ''' class to control camera over the mil frame grabber'''
    DEFAULT = {'name': 'milCamera',
               'exposureTime': 10, # ms initially automatically set the exposure time
               'nFrame': 1,
               'n_buffer_save': 2**3, # number of buffered images on the grabber card
               'roiHeight': None, # rows to keep (centered crop), None = full sensor height
               'roiOffsetY': None, # first row of the crop, None = centered within the sensor
    }


    def __init__(self, name=None, **kwargs):

        if name is None: name=MilCamera.DEFAULT['name'] 
        super().__init__(name=name,**kwargs)

        # Mil parameters
        self.MilApplication = None
        self.MilSystem = None
        self.MilDisplay = None
        self.MilDigitizer = None 

        #self.GrabBuffer = []
        self.SaveBuffer = []

        ## Standard, don't change
        self.ExposureTime_0 = 999850
        self.AcquisitionFrameRate_0 = 1

        #### Measurement parameters
        self.n_buffer_save = MilCamera.DEFAULT['n_buffer_save']
        # camera parameters
        self.exposureTime = MilCamera.DEFAULT['exposureTime']
        self.nFrame = MilCamera.DEFAULT['nFrame']
        # only cropping rows speeds up acquisition on this sensor (columns
        # are read out in parallel, and binning was measured to not reduce
        # readout time either - only windowing the number of rows does);
        # set before connect() - it determines the buffer/image dimensions
        self.roiHeight = MilCamera.DEFAULT['roiHeight']
        # first row of the roiHeight crop; None centers it. Ignored when
        # roiHeight is None (nothing to offset within the full sensor).
        self.roiOffsetY = MilCamera.DEFAULT['roiOffsetY']



    def connect(self):
        super().connect()
        self.prepareCamera()

        self.setParameter('exposureTime',self.exposureTime)

    def prepareCamera(self):
        ''' prepare camera and make initial setting '''

        # Allocate defaults
        self.MilApplication, self.MilSystem, self.MilDisplay, self.MilDigitizer = MIL.MappAllocDefault(MIL.M_DEFAULT, ImageBufIdPtr=MIL.M_NULL)

        # Vertical ROI (row windowing), set before querying the image size
        # below so buffers get sized to the (possibly reduced) height.
        # Height/OffsetY/Binning persist on the camera itself across
        # connect()/disconnect() cycles (MappFreeDefault/MappAllocDefault
        # don't reset them), so always reset to a known-clean state first -
        # that's what makes roiHeight=None reliably mean "full sensor
        # height" rather than "whatever a previous session left behind".
        # Binning in particular must be reset before querying Height's max:
        # this camera reports SensorHeight/HeightMax relative to the
        # current BinningVertical (e.g. 3002 instead of 6004 at binning=2),
        # so a leftover binning setting would silently halve the "full
        # height" baseline. (BinningVertical/Horizontal and
        # DecimationVertical were tried for speed first: binning changed
        # the image size but not the readout time, and this camera has no
        # decimation feature at all (confirmed via CamExpert's
        # ImageFormatControl feature list) - only Height/OffsetY actually
        # move the frame-rate ceiling here.)
        _BinningHorizontalReset = MIL.MIL_INT(1)
        MIL.MdigControlFeature(self.MilDigitizer, MIL.M_FEATURE_VALUE, MIL.MIL_TEXT("BinningHorizontal"), MIL.M_TYPE_MIL_INT, ctypes.byref(_BinningHorizontalReset))
        _BinningVerticalReset = MIL.MIL_INT(1)
        MIL.MdigControlFeature(self.MilDigitizer, MIL.M_FEATURE_VALUE, MIL.MIL_TEXT("BinningVertical"), MIL.M_TYPE_MIL_INT, ctypes.byref(_BinningVerticalReset))

        _OffsetYReset = MIL.MIL_INT(0)
        MIL.MdigControlFeature(self.MilDigitizer, MIL.M_FEATURE_VALUE, MIL.MIL_TEXT("OffsetY"), MIL.M_TYPE_MIL_INT, ctypes.byref(_OffsetYReset))

        maxHeight = MIL.MdigInquireFeature(self.MilDigitizer, MIL.M_FEATURE_MAX, "Height", MIL.M_TYPE_MIL_INT)
        _HeightFull = MIL.MIL_INT(int(maxHeight))
        MIL.MdigControlFeature(self.MilDigitizer, MIL.M_FEATURE_VALUE, MIL.MIL_TEXT("Height"), MIL.M_TYPE_MIL_INT, ctypes.byref(_HeightFull))

        if self.roiHeight is not None:
            if self.roiOffsetY is not None:
                offsetY = self.roiOffsetY
            else:
                offsetY = (maxHeight - self.roiHeight) // 2

            _Height = MIL.MIL_INT(int(self.roiHeight))
            MIL.MdigControlFeature(self.MilDigitizer, MIL.M_FEATURE_VALUE, MIL.MIL_TEXT("Height"), MIL.M_TYPE_MIL_INT, ctypes.byref(_Height))

            _OffsetY = MIL.MIL_INT(int(offsetY))
            MIL.MdigControlFeature(self.MilDigitizer, MIL.M_FEATURE_VALUE, MIL.MIL_TEXT("OffsetY"), MIL.M_TYPE_MIL_INT, ctypes.byref(_OffsetY))

        readHeight = MIL.MdigInquireFeature(self.MilDigitizer, MIL.M_FEATURE_VALUE, "Height", MIL.M_TYPE_MIL_INT)
        readOffsetY = MIL.MdigInquireFeature(self.MilDigitizer, MIL.M_FEATURE_VALUE, "OffsetY", MIL.M_TYPE_MIL_INT)
        readBinningVertical = MIL.MdigInquireFeature(self.MilDigitizer, MIL.M_FEATURE_VALUE, "BinningVertical", MIL.M_TYPE_MIL_INT)
        readBinningHorizontal = MIL.MdigInquireFeature(self.MilDigitizer, MIL.M_FEATURE_VALUE, "BinningHorizontal", MIL.M_TYPE_MIL_INT)
        logger.info(f'roiHeight requested {self.roiHeight}, roiOffsetY requested {self.roiOffsetY} '
                    f'(sensor max height {maxHeight}), camera reports '
                    f'Height={readHeight}, OffsetY={readOffsetY}, BinningVertical={readBinningVertical}, BinningHorizontal={readBinningHorizontal}')

        # image parameters
        self.height = MIL.MdigInquire(self.MilDigitizer, MIL.M_SIZE_Y)
        self.width = MIL.MdigInquire(self.MilDigitizer, MIL.M_SIZE_X)

        # Allocate the save buffers (grab ring, for double-buffering)
        for n in range(0, self.n_buffer_save):
            self.SaveBuffer.append(
                MIL.MbufAlloc2d(self.MilSystem,
                MIL.MdigInquire(self.MilDigitizer, MIL.M_SIZE_X),
                MIL.MdigInquire(self.MilDigitizer, MIL.M_SIZE_Y),
                16 + MIL.M_UNSIGNED,
                MIL.M_IMAGE + MIL.M_GRAB + MIL.M_PROC))
            MIL.MbufClear(self.SaveBuffer[n], MIL.M_COLOR_BLACK);

        # Separate on-board accumulator (same 16-bit depth as SaveBuffer, so
        # MimArith adds are plain value-preserving sums - no cross-depth
        # rescaling). Never used as a grab target, so the MdigProcess hook
        # can safely add into it between host flushes.
        self.GroupAccBuffer = MIL.MbufAlloc2d(self.MilSystem,
            MIL.MdigInquire(self.MilDigitizer, MIL.M_SIZE_X),
            MIL.MdigInquire(self.MilDigitizer, MIL.M_SIZE_Y),
            16 + MIL.M_UNSIGNED,
            MIL.M_IMAGE + MIL.M_PROC)
        MIL.MbufClear(self.GroupAccBuffer, MIL.M_COLOR_BLACK);

    def disconnect(self):
        self.free_alloc()
        super().disconnect()


    def _setExposureTime(self, value):
        ''' ExposureTime in miliseconds '''

        ## just change exposure time
        self.exposureTime = value
        exposureTime_um = 1000* self.exposureTime

        # AcquisitionFrameRate/AcquisitionFramePeriod persist on the camera
        # across connect()/disconnect() cycles, same as Height/OffsetY/
        # Binning - a stale leftover value (e.g. from an earlier, unrelated
        # ROI/session) silently caps the frame rate regardless of the
        # current ExposureTime/Height, so both must be explicitly (re)set
        # every time rather than left alone. (A same-session A/B test can
        # look like "no difference" here even when it matters, because the
        # untouched run coasts on whatever the previous run just set -
        # always test against a fresh/leftover-contaminated baseline, not
        # a same-session neighbor.) Request the fastest achievable cycle:
        # frame rate scaled inversely from the ExposureTime_0/
        # AcquisitionFrameRate_0 baseline, and frame period requested equal
        # to the exposure time itself (us, matching ExposureTime's unit).
        # The camera clamps both to whatever it can actually sustain.
        self.AcquisitionFrameRate = MIL.MIL_INT(int(np.floor(self.AcquisitionFrameRate_0 * self.ExposureTime_0 /exposureTime_um) ))
        self.AcquisitionFramePeriod = MIL.MIL_INT(exposureTime_um)

        _ExposureTime = MIL.MIL_INT(exposureTime_um)

        # Put the digitizer in asynchronous mode to be able to process while grabbing.
        MIL.MdigControl(self.MilDigitizer, MIL.M_GRAB_MODE, MIL.M_ASYNCHRONOUS)

        MIL.MdigControlFeature(self.MilDigitizer, MIL.M_FEATURE_VALUE, MIL.MIL_TEXT("AcquisitionFrameRate"), MIL.M_TYPE_MIL_INT, ctypes.byref(self.AcquisitionFrameRate))
        MIL.MdigControlFeature(self.MilDigitizer, MIL.M_FEATURE_VALUE, MIL.MIL_TEXT("AcquisitionFramePeriod"), MIL.M_TYPE_MIL_INT, ctypes.byref(self.AcquisitionFramePeriod))
        MIL.MdigControlFeature(self.MilDigitizer, MIL.M_FEATURE_VALUE, MIL.MIL_TEXT("ExposureTime"), MIL.M_TYPE_MIL_INT, ctypes.byref(_ExposureTime))
        
    def free_alloc(self):
        ''' free the buffer on the mil '''
        for n in range(0, self.n_buffer_save):
            MIL.MbufFree(self.SaveBuffer[n])
        MIL.MbufFree(self.GroupAccBuffer)

        MIL.MappFreeDefault(self.MilApplication, self.MilSystem, self.MilDisplay, self.MilDigitizer, MIL.M_NULL)


    def getLastImage(self):
        ''' grab self.nFrame frames and average them.

        Individual synchronous MdigGrab/MdigGrabWait calls were measured to
        cost ~2x the camera's actual frame period - issuing grabs one at a
        time does not let the hardware run continuously. MdigProcess keeps
        the digitizer continuously acquiring into a ring of buffers and
        invokes a hook per frame instead, which lets it sustain close to
        its real frame-rate ceiling. Each frame is summed on-board and only
        flushed to the host every n_buffer_save frames (see
        _AccumulateHookData/_accumulateFrameHook), so the hook stays fast
        enough not to fall behind the grab rate. '''

        nFrame = self.nFrame
        nBuf = self.n_buffer_save

        hookData = _AccumulateHookData(nFrame, self.GroupAccBuffer, nBuf, self.height, self.width)
        hookFunctionPtr = MIL.MIL_DIG_HOOK_FUNCTION_PTR(_accumulateFrameHook)

        MIL.MbufClear(self.GroupAccBuffer, MIL.M_COLOR_BLACK)

        tStart = time.perf_counter()
        MIL.MdigProcess(self.MilDigitizer, self.SaveBuffer, nBuf, MIL.M_START, MIL.M_DEFAULT, hookFunctionPtr, hookData)

        timedOut = not hookData.done.wait(timeout=max(5.0, nFrame * 0.5))

        MIL.MdigProcess(self.MilDigitizer, self.SaveBuffer, nBuf, MIL.M_STOP, MIL.M_DEFAULT, hookFunctionPtr, hookData)
        hookData.flush()
        tTotal = time.perf_counter() - tStart

        if timedOut:
            logger.warning(f'getLastImage: timed out waiting for {nFrame} frames, only got {hookData.count}')

        processFrameCount = MIL.MdigInquire(self.MilDigitizer, MIL.M_PROCESS_FRAME_COUNT)
        processFrameRate = MIL.MdigInquire(self.MilDigitizer, MIL.M_PROCESS_FRAME_RATE)

        self.rawImage = hookData.total / max(hookData.count, 1)

        logger.info(f'getLastImage({nFrame} frames, {hookData.count} received): total {tTotal*1000:.1f} ms | '
                    f'MIL-reported {processFrameCount} frames at {processFrameRate:.1f} fps')

        return self.rawImage



if __name__ == "__main__":
    cam = MilCamera()
    cam.connect()
    cam._displayStreamOfImages()

    cam.disconnect()




# %%
