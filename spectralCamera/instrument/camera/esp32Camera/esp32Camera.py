"""
Camera DeviceModel

@author: ostranik
"""
#%%

import os
import time
import subprocess
import tempfile
import threading
import numpy as np
import cv2
import requests
from viscope.instrument.base.baseCamera import BaseCamera

class ESP32Camera(BaseCamera):
    ''' class to control an ESP32-CAM board running the webCamCom firmware
    (see https://github.com/ondrejstranik/webCamCom).
    the board hosts a control server on http://<ip>:<controlPort>/camera and
    streams MJPEG frames on http://<ip>:<streamPort>/stream.
    when the board cannot join an existing WiFi network it starts its own
    access point, with IP Gateway 192.168.4.1 (the default here) '''

    DEFAULT = {'name': 'esp32Camera',
                'ssid': 'esp32', # wifi access point of the esp32cam (open, no password)
                'ip': '192.168.4.1', # IP of the esp32cam (AP gateway IP by default)
                'controlPort': 80,
                'streamPort': 81,
                'nFrame': 1,
                'filterType': 'RGB', # type of the filter 'RGB', 'RGGB', 'BW'
                'framesize': 6, # esp32cam framesize_t index (6 = FRAMESIZE_CIF, 400x296)
                'quality': 15} # jpeg compression quality, 1 (best) - 63 (worst)

    def __init__(self, name=None,*args,**kwargs):
        ''' initialisation '''

        if name is None: name=ESP32Camera.DEFAULT['name']
        super().__init__(name=name,**kwargs)

        # camera parameters
        self.ssid = kwargs['ssid'] if 'ssid' in kwargs else ESP32Camera.DEFAULT['ssid']
        self.ip = kwargs['ip'] if 'ip' in kwargs else ESP32Camera.DEFAULT['ip']
        self.controlPort = kwargs['controlPort'] if 'controlPort' in kwargs else ESP32Camera.DEFAULT['controlPort']
        self.streamPort = kwargs['streamPort'] if 'streamPort' in kwargs else ESP32Camera.DEFAULT['streamPort']
        if 'filterType' in kwargs:
            self.filterType = kwargs['filterType']
        elif 'rgbOrder' in kwargs:
            self.filterType = kwargs['rgbOrder']
        else:
            self.filterType = ESP32Camera.DEFAULT['filterType']
        self.framesize = kwargs['framesize'] if 'framesize' in kwargs else ESP32Camera.DEFAULT['framesize']
        self.quality = kwargs['quality'] if 'quality' in kwargs else ESP32Camera.DEFAULT['quality']

        self.exposureTime = None
        self.nFrame = ESP32Camera.DEFAULT['nFrame']

        self._streamResponse = None
        self._streamIterator = None
        self._streamBuffer = b''

        # wifi network the PC was connected to before switching to the esp32cam's
        # access point, so it can be restored again on disconnect()
        self._previousWifi = None

        # background reader thread, always keeping only the most recently
        # decoded frame, so getLastImage() never lags behind a backlog of
        # already-received but unconsumed frames
        self._readerThread = None
        self._readerStop = False
        self._latestFrame = None
        self._frameLock = threading.Lock()
        self._frameEvent = threading.Event()

    @property
    def controlUrl(self):
        ''' url of the camera control server (quality, exposure, gain, ...) '''
        return f'http://{self.ip}:{self.controlPort}/camera'

    @property
    def streamUrl(self):
        ''' url of the MJPEG video stream '''
        return f'http://{self.ip}:{self.streamPort}/stream'

    def connect(self):
        super().connect()

        # join the esp32cam's own wifi access point (open network, no password)
        self._connectWifi()

        # the esp32cam MJPEG stream is a multipart/x-mixed-replace http response
        # with a non-standard boundary placement (trailing instead of leading);
        # cv2.VideoCapture hangs indefinitely on it, so the stream bytes are
        # parsed manually, looking for the raw JPEG start/end markers
        self._streamResponse = requests.get(self.streamUrl, stream=True, timeout=5)
        self._streamIterator = self._streamResponse.iter_content(chunk_size=1024)
        self._streamBuffer = b''

        # continuously drain the stream in the background, so a slow consumer
        # never builds up a backlog of stale, already-received frames
        self._readerStop = False
        self._readerThread = threading.Thread(target=self._streamReaderLoop,daemon=True)
        self._readerThread.start()

        # switch off auto-exposure/gain/white-balance for a stable, calibrated signal
        self._setParameterESP32('allAuto', 0)

        # set the default image size and jpeg compression quality
        self._setParameterESP32('framesize', self.framesize)
        self._setParameterESP32('quality', self.quality)

        # get the image size from the first frame
        temporary_frame = None
        while temporary_frame is None:
            temporary_frame = self._readFrame()
        self.height, self.width = temporary_frame.shape[0:2]
        if self.filterType == 'RGB':
            self.width *= 3
        if self.filterType == 'RGGB':
            self.width *= 2
            self.height *= 2

        # get the camera current exposure
        self.exposureTime = self.getParameter('exposureTime')

    def disconnect(self):
        super().disconnect()
        self._readerStop = True
        self._streamResponse.close()
        if self._readerThread is not None:
            self._readerThread.join(timeout=2)

        # reconnect the PC to whatever wifi network it was on before
        self._reconnectPreviousWifi()

    def __str__(self):
        return f'ESP32 Camera {self.ip}'

    def getLastImage(self):
        myframe = None
        for _ in range(self.nFrame):
            temporary_frame = None
            while temporary_frame is None:
                temporary_frame = self._readFrame()

            if self.filterType == 'RGB':
                if myframe is None:
                    myshape = np.shape(temporary_frame.T)
                    myframe = np.reshape(temporary_frame.T, (myshape[0]*myshape[1], myshape[2]))
                    myframe = myframe.astype('int64').T
                else:
                    myframe = myframe + np.reshape(temporary_frame.T, (myshape[0]*myshape[1], myshape[2])).T

            if self.filterType == 'RGGB':
                _myframe = np.empty((temporary_frame.shape[0]*2,temporary_frame.shape[1]*2))
                _myframe[0::2,0::2] = temporary_frame[:,:,0] #R
                _myframe[0::2,1::2] = temporary_frame[:,:,1] //2 #G
                _myframe[1::2,0::2] = temporary_frame[:,:,1] //2 #G
                _myframe[1::2,1::2] = temporary_frame[:,:,2] //2 #B
                if myframe is None:
                    myframe = _myframe.astype('int64')
                else:
                    myframe = myframe + _myframe

            if self.filterType == 'BW':
                _myframe = np.sum(temporary_frame,axis=2)
                if myframe is None:
                    myframe = _myframe.astype('int64')
                else:
                    myframe = myframe + _myframe

        self.rawImage = myframe/self.nFrame
        return self.rawImage

    def _streamReaderLoop(self):
        ''' background loop: continuously decode frames from the MJPEG http
        stream as fast as the network delivers them, always overwriting
        the single most-recently-decoded frame (never queuing a backlog).
        if the connection itself breaks (wifi hiccup, esp32cam stall, ...) it
        is automatically reopened - never spin-retry without backoff, that
        pegs a cpu core and starves the whole application '''
        while not self._readerStop:
            try:
                frame = self._decodeNextFrame()
            except Exception as error:
                if self._readerStop:
                    break
                print(f'esp32cam stream error ({error}), reconnecting...')
                self._reopenStream()
                continue

            if frame is None:
                continue
            with self._frameLock:
                self._latestFrame = frame
            self._frameEvent.set()

    def _reopenStream(self,retryDelay=1.0):
        ''' close and reopen the http connection to the esp32cam MJPEG stream,
        retrying with a delay between attempts until it succeeds or
        disconnect() requests the reader to stop '''
        try:
            self._streamResponse.close()
        except Exception:
            pass
        self._streamBuffer = b''

        while not self._readerStop:
            try:
                self._streamResponse = requests.get(self.streamUrl, stream=True, timeout=5)
                self._streamIterator = self._streamResponse.iter_content(chunk_size=1024)
                return
            except requests.exceptions.RequestException:
                time.sleep(retryDelay)

    def _decodeNextFrame(self):
        ''' pull bytes from the MJPEG http stream until one full JPEG frame
        (delimited by its own start-of-image/end-of-image markers) is decoded.
        returns None if no full frame is available yet.
        raises if the underlying http stream itself failed '''
        self._streamBuffer += next(self._streamIterator)

        start = self._streamBuffer.find(b'\xff\xd8')
        if start == -1:
            self._streamBuffer = b''
            return None

        end = self._streamBuffer.find(b'\xff\xd9', start+2)
        if end == -1:
            # discard any garbage before the jpeg start, keep waiting for the end marker
            self._streamBuffer = self._streamBuffer[start:]
            return None

        jpgBytes = self._streamBuffer[start:end+2]
        self._streamBuffer = self._streamBuffer[end+2:]
        return cv2.imdecode(np.frombuffer(jpgBytes,dtype=np.uint8), cv2.IMREAD_COLOR)

    def _readFrame(self):
        ''' block until the background reader has a new decoded frame ready,
        then consume and return it (always the latest one available) '''
        self._frameEvent.wait()
        with self._frameLock:
            frame = self._latestFrame
            self._latestFrame = None
            self._frameEvent.clear()
        return frame

    def _setExposureTime(self,value):
        ''' set the exposure of the esp32cam sensor.
        value is in the sensor's own AEC units (roughly 0-1200), not milliseconds:
        the esp32cam sensor does not expose a time-calibrated exposure register '''

        value = int(np.clip(value,0,1200))
        self._setParameterESP32('exposure', value)
        self.exposureTime = value

    def _getExposureTime(self):
        _status = self._getStatusESP32()
        self.exposureTime = _status['exposure']
        return self.exposureTime

    def _connectWifi(self,maxAttempt=20):
        ''' connect the PC's wifi adapter to the esp32cam access point.
        the access point is open (no password), broadcast under self.ssid.
        windows only (uses netsh) '''

        # remember the currently connected network, to restore it on disconnect()
        self._previousWifi = self._getCurrentWifiName()

        profileXml = (
            '<?xml version="1.0"?>'
            '<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">'
            f'<name>{self.ssid}</name>'
            f'<SSIDConfig><SSID><name>{self.ssid}</name></SSID></SSIDConfig>'
            '<connectionType>ESS</connectionType>'
            '<connectionMode>manual</connectionMode>'
            '<MSM><security><authEncryption>'
            '<authentication>open</authentication>'
            '<encryption>none</encryption>'
            '<useOneX>false</useOneX>'
            '</authEncryption></security></MSM>'
            '</WLANProfile>'
        )

        profilePath = os.path.join(tempfile.gettempdir(), f'{self.ssid}.xml')
        with open(profilePath,'w') as profileFile:
            profileFile.write(profileXml)

        subprocess.run(['netsh','wlan','add','profile',f'filename={profilePath}','user=all'],
                        capture_output=True)
        subprocess.run(['netsh','wlan','connect',f'name={self.ssid}',f'ssid={self.ssid}'],
                        capture_output=True)

        # wait until the esp32cam is reachable over the new wifi connection
        for _ in range(maxAttempt):
            try:
                if requests.get(f'http://{self.ip}:{self.controlPort}/ping', timeout=1).ok:
                    print(f'connected to wifi {self.ssid}, esp32cam reachable at {self.ip}')
                    return
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)

        raise ConnectionError(f'could not reach the esp32cam at {self.ip} on wifi {self.ssid}')

    def _getCurrentWifiName(self):
        ''' name of the wifi network the PC is currently connected to (its SSID,
        which by default is also the saved profile name), or None if not
        connected to any wifi network. windows only (uses netsh) '''
        result = subprocess.run(['netsh','wlan','show','interfaces'],
                                capture_output=True, text=True)
        for line in result.stdout.splitlines():
            stripped = line.strip()
            # match the 'SSID' line but not 'BSSID'
            if stripped.startswith('SSID') and not stripped.startswith('BSSID'):
                _, _, value = stripped.partition(':')
                value = value.strip()
                if value:
                    return value
        return None

    def _reconnectPreviousWifi(self):
        ''' reconnect the PC to the wifi network it was on before connect()
        switched it to the esp32cam's access point. does nothing if there
        was none (e.g. the PC was not on any wifi network before) '''
        if self._previousWifi is None:
            return
        subprocess.run(['netsh','wlan','connect',f'name={self._previousWifi}'],
                        capture_output=True)
        print(f'reconnecting to previous wifi {self._previousWifi}')

    def _setParameterESP32(self, parameter, value):
        ''' set a parameter of the esp32cam over http
        http://<ip>:<controlPort>/camera?set=<parameter>&value=<value>
        parameter is one of quality, ae, exposure, gain, brightness, contrast,
        saturation, framesize, allAuto (see webCamCom/src/main.cpp) '''
        requests.get(self.controlUrl, params={'set': parameter, 'value': value}, timeout=1)

    def _getStatusESP32(self):
        ''' get the full camera status (quality, exposure, gain, ...) as a dict '''
        response = requests.get(self.controlUrl, timeout=1)
        return response.json()


#%%

if __name__ == '__main__':
    from spectralCamera.instrument.camera.esp32Camera.esp32Camera import ESP32Camera

    cam = ESP32Camera(name='ESP32Camera',filterType='RGB')
    cam.connect()
    cam.setParameter('exposureTime',1)
    cam.setParameter('nFrames', 1)

    cam._displayStreamOfImages()
    cam.disconnect()
