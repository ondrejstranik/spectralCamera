'''
interface for photonfocus camera 
'''
#%% link the library
import os, sys
sys.path.append(os.path.join(os.environ['PF_ROOT'],'PFSDK','bin'))
if sys.version_info >= (3,8):
    os.add_dll_directory(os.path.join(os.environ['PF_ROOT'],'DoubleRateSDK','bin'))

import PFPyCameraLib as pf

import numpy as np
import time

#%%

def PromptEnterAndExit(code = 0):
    input("\n\n\nPress enter to exit ...")
    sys.exit(code)

def ExitWithErrorPrompt(errString, pfResult = None):
    print(errString)
    if pfResult is not None:
        print(pfResult)
    PromptEnterAndExit(-1)

def EventErrorCallback(cameraNumber, errorCode, errorMessage):
    print("[Communication error callback] Camera(",cameraNumber,") Error(", errorCode, ", ", errorMessage, ")\n")


class Photonfocus:
    def __init__(self):
        ''' set the initial parameters of the camera '''

        # camera related recording parameter
        self.cam_info = None
        self.pfCam = None
        self.pfStream = None
        self.pfBuffer = None
        self.pfBufferReleased = True
        self.ringSizeBuffer = 2

        # camera type related spectral parameter 
        self.pixelChar = {}
        self.defaultCalibrationFile = os.path.join(
            os.path.dirname(__file__), r"CMV2K-SSM5x5-600_1000-5.6.6.11.xml")
        # if self.calibrationFile = 'plain' then no spectral calibration is done
        self.calibrationFile = self.defaultCalibrationFile
        # matrix index = 0 for filter with 600-875nm band pass
        # matrix index = 1 for filter with 675-975nm band pass        
        self.matrix_index = 0
        self.chipOffsetTopLeft = [0,3]
        self.chipOffsetBottomRight = [0,5]

        self.darkImage = 0

        # default parameters        
        self.cameraIdx = 0
        self.pixelFormat = "Mono8"
        self.exposureTime_um = 100000  
        self.height = None
        self.width = None

        # set the call back function
        pf.SetEventErrorReceiver(EventErrorCallback)

    def SelectCamera(self,cameraIdx=None):
        ''' select camera
        if cameraIdx is None, user input is needed
        '''
        
        if cameraIdx:
            self.cameraIdx = cameraIdx
        
        #Discover cameras in the network or connected to the USB port
        discovery = pf.PFDiscovery()
        pfResult = discovery.DiscoverCameras()

        if pfResult != pf.Error.NONE:
            ExitWithErrorPrompt("Discovery error:", pfResult)

        if self.cameraIdx is None:
            #Print all available cameras
            num_discovered_cameras = discovery.GetCameraCount()
            camera_info_list = []
            for x in range(num_discovered_cameras):
                [pfResult, camera_info] = discovery.GetCameraInfo(x)
                camera_info_list.append(camera_info) 
                print("[",x,"]")
                print(camera_info_list[x])

            #Prompt user to select a camera
            user_input = input("Select camera: ")
            try:
                cam_id = int(user_input)
            except:
                ExitWithErrorPrompt("Error parsing input, not a number")

            #Check selected camera is within range
            if not 0 <= cam_id < num_discovered_cameras:
                ExitWithErrorPrompt("Selected camera out of range")
            
            selected_cam_info = camera_info_list[cam_id]
        else:
            [pfResult, selected_cam_info] = discovery.GetCameraInfo(int(self.cameraIdx))


        #Call copy constructor
        #The camera info list elements are destroyed with PFDiscover
        if selected_cam_info.GetType() == pf.CameraType.CAMTYPE_GEV:
            copy_cam_info = pf.PFCameraInfoGEV(selected_cam_info)
        else:
            copy_cam_info = pf.PFCameraInfoU3V(selected_cam_info)

        self.cam_info = copy_cam_info

    def ConnectCamera(self):
        #Connect camera
        pfCam = pf.PFCamera()
        pfResult = pfCam.Connect(self.cam_info)
        #pfResult = pfCam.Connect(ip = "192.168.3.158")
        if pfResult != pf.Error.NONE:
            ExitWithErrorPrompt("Could not connect to the selected camera", pfResult)

        #This sample assumes no DR
        #But to ensure it works with all cameras, we are going to disable DR
        [pfResult, featureList] = pfCam.GetFeatureList()
        if pfResult != pf.Error.NONE:
            ExitWithErrorPrompt("Could not get feature list from camera", pfResult)

        #Check DoubleRate_Enable feature is present
        if any(elem.Name == "DoubleRate_Enable" for elem in featureList):
            print("DoubleRate_Enable feature found. Disabling feature.")
            pfResult = pfCam.SetFeatureBool("DoubleRate_Enable", False)
            if pfResult != pf.Error.NONE:
                ExitWithErrorPrompt("Failed to set DoubleRate_Enable", pfResult)

        self.pfCam = pfCam

        # Set up the stream
        self._SetupStream()

    def PrepareCamera(self,cameraIdx=None):
        if cameraIdx:
            self.cameraIdx = cameraIdx
        self.SelectCamera()
        self.ConnectCamera()
        self.SetDefaultParameters()

    def SetFOVToMax(self):
        ''' set field of view to maximal size '''
        #Set Width to maximum
        pfResult, pfFeatureParam = self.pfCam.GetFeatureParams("Width")
        if pfResult != pf.Error.NONE:
            ExitWithErrorPrompt("Could not get width feature parameters", pfResult)
        
        self.width = pfFeatureParam.Max
        self.SetParameter("Width", self.width)

        #Set Height to maximum
        pfResult, pfFeatureParam = self.pfCam.GetFeatureParams("Height")
        if pfResult != pf.Error.NONE:
            ExitWithErrorPrompt("Could not get Height feature parameters", pfResult)

        self.height = pfFeatureParam.Max
        self.SetParameter("Height", self.height)

    def SetSpectralFOVtoMax(self):
        ''' set field of view of full superpixel to max '''

        # set Width
        pfResult, pfFeatureParam = self.pfCam.GetFeatureParams("Width")
        if pfResult != pf.Error.NONE:
            ExitWithErrorPrompt("Could not get width feature parameters", pfResult)
        
        self.width = pfFeatureParam.Max - self.chipOffsetBottomRight[0] - self.chipOffsetTopLeft[0]
        self.SetParameter("Width", self.width)
        self.SetParameter("OffsetX", self.chipOffsetTopLeft[0])


        #Set Height
        pfResult, pfFeatureParam = self.pfCam.GetFeatureParams("Height")
        if pfResult != pf.Error.NONE:
            ExitWithErrorPrompt("Could not get Height feature parameters", pfResult)

        self.height = pfFeatureParam.Max - self.chipOffsetBottomRight[1] - self.chipOffsetTopLeft[1]
        self.SetParameter("Height", self.height)
        self.SetParameter("OffsetY", self.chipOffsetTopLeft[1])


    def SetDefaultParameters(self):
        ''' set default parameters of the camera '''
        #Set pixel format
        self.SetParameter("PixelFormat",self.pixelFormat)

        # set Exposure
        self.SetParameter("ExposureTime",self.exposureTime_um)

        # set field of view to max hyperspectral pixels
        self.SetSpectralFOVtoMax()

    def SetParameter(self,parameter,value):
        # set the camera parameters 
        pfResult, typeNumber = self.pfCam.GetFeatureType(parameter)
        if pfResult != pf.Error.NONE:
            ExitWithErrorPrompt("Could not get feature type", pfResult)

        # set the parameter
        if typeNumber==7: # bool
            pfResult = self.pfCam.SetFeatureBool(parameter, value)
        elif typeNumber==4: #str
            pfResult = self.pfCam.SetFeatureEnum(parameter, value)
        elif typeNumber==3: # float
            pfResult = self.pfCam.SetFeatureFloat(parameter,value)
        elif typeNumber==2: # integer
            pfResult = self.pfCam.SetFeatureInt(parameter, value)
        elif typeNumber==5: # enumeration
            pfResult = self.pfCam.SetFeatureEnum(parameter, value)

        if pfResult and pfResult != pf.Error.NONE:
            ExitWithErrorPrompt(f"Could not set {parameter}", pfResult)

    def GetParameter(self,parameter:str):
        # return the value of the parameter
        pfResult, typeNumber = self.pfCam.GetFeatureType(parameter)
        if pfResult != pf.Error.NONE:
            ExitWithErrorPrompt("Could not get feature type", pfResult)

        if typeNumber==7: # bool
            pfResult, value = self.pfCam.GetFeatureBool(parameter)
        elif typeNumber==4: #str
            pfResult, value = self.pfCam.GetFeatureString(parameter)
        elif typeNumber==3: # float
            pfResult, value = self.pfCam.GetFeatureFloat(parameter)
        elif typeNumber==2: # integer
            pfResult, value = self.pfCam.GetFeatureInt(parameter)
        elif typeNumber==5: # enumeration
            pfResult, value = self.pfCam.GetFeatureEnum(parameter)
        if pfResult and pfResult != pf.Error.NONE:
            ExitWithErrorPrompt("Could not get the value", pfResult)

        return value

    def GetPlainCalibrationData(self):
        ''' pure single pixel values are used to get the hyperspectral data '''

        if self.matrix_index==0:
            self.pixelChar['wv'] = np.linspace(600,875,25)
        else:
            self.pixelChar['wv'] = np.linspace(675,975,25)
        self.pixelChar['coef_pixels'] = np.eye(25)
        self.pixelChar['filterchar'] = self.pixelChar['wv']*0 + 1 

        self.calibrationFile = 'plain'

    def GetCalibrationData(self,calibrationFile=None):
        ''' get the calibration data to calculate the Hyper spectral data '''
        from lxml import etree

        if calibrationFile:
            self.calibrationFile = calibrationFile
        else:
            self.calibrationFile = self.defaultCalibrationFile

        try:
            tree = etree.parse(self.calibrationFile)

            root = tree.getroot()
            w1 = float(root.findall("./filter_info/calibration_info/wavelength_range_start_nm")[0].text)
            w2 = float(root.findall("./filter_info/calibration_info/wavelength_range_end_nm")[0].text)
            wres = float(root.findall("./filter_info/calibration_info/wavelength_resolution_nm")[0].text)
            w = np.arange(w1,w2+wres,wres)

            spec_char = root.findall("./filter_info/filter_zones/filter_zone/bands/band/response")
            
            filterchar = np.array([])
            for ii in range(len(spec_char)):
                f1 = np.fromstring(spec_char[ii].text, sep = ',')
                filterchar = np.append(filterchar,f1)
            filterchar = np.reshape(filterchar,(len(spec_char),-1)).T
            
            c_matrix = root.findall("./system_info/spectral_correction_info/correction_matrices/correction_matrix")

            w_vir = c_matrix[self.matrix_index].findall('./virtual_bands/virtual_band/wavelength_nm')
            fwhm_vir = c_matrix[self.matrix_index].findall('./virtual_bands/virtual_band/fwhm_nm')
            coef_vir = c_matrix[self.matrix_index].findall('./virtual_bands/virtual_band/coefficients')

            coef_pixels = np.array([])
            for ii in range(len(coef_vir)):
                f1 = np.fromstring(coef_vir[ii].text, sep = ',')
                coef_pixels = np.append(coef_pixels,f1)
            coef_pixels = np.reshape(coef_pixels,(len(coef_vir),-1)).T

            wv = np.array([float(i.text) for i in w_vir])
            fwhmv = np.array([float(i.text) for i in fwhm_vir])

            self.pixelChar['w'] = w
            self.pixelChar['filterchar'] = filterchar
            self.pixelChar['wv'] = wv
            self.pixelChar['fwhmv'] = fwhmv
            self.pixelChar['coef_pixels'] = coef_pixels

        except:
            print(f'could not read the file {self.calibrationFile}')
            print(f'switching to plain pixel values')
            self.GetPlainCalibrationData()

    def plotCalibrationData():
        ''' plot the calibration data '''

        import matplotlib.pyplot as plt 

        # renaming
        w = self.pixelChar['w']
        filterchar = self.pixelChar['filterchar'] 
        wv = self.pixelChar['wv'] 
        fwhmv = self.pixelChar['fwhmv'] 
        coef_pixels = self.pixelChar['coef_pixels']

        # ploting 
        f, ax = plt.subplots(3,1)
        ax[0].errorbar(np.arange(len(wv)),wv, yerr = fwhmv/2)
        ax[0].set_xlabel('channel in super-pixel')
        ax[0].set_ylabel('( wavelength ;  errorbar = FWHM ) / nm')

        #ax[1].plot(np.linspace(w1,w2,filterchar.shape[0]),filterchar)
        ax[1].plot(w,filterchar)
        ax[1].set_xlabel('wavelength / nm')
        ax[1].set_xlim(w[0],w[-1])
        ax[1].set_ylabel('Transmission @ given channel')

        ax[2].imshow(filterchar.T, aspect = 'auto')
        ax[2].set_xlabel('wavelength / nm')
        ax[2].set_xticks(w[::40]-w[0])
        ax[2].set_xticklabels(w[::40].astype(np.int16))
        ax[2].set_ylabel('channel')
        f.suptitle('Super pixel spectral characterisation')

        f, ax = plt.subplots(1,1)
        ax.imshow(coef_pixels)
        f.suptitle('Spectral de-convolution matrix')
        plt.show()

    def _SetupStream(self,ringSizeBuffer=None):
        #Create stream depending on camera type
        if self.cam_info.GetType() == pf.CameraType.CAMTYPE_GEV:
            self.pfStream = pf.PFStreamGEV(False, True, False, True)
        else:
            self.pfStream = pf.PFStreamU3V()
        
        #Set ring buffer size 
        if ringSizeBuffer:
            self.ringSizeBuffer = ringSizeBuffer

        self.pfStream.SetBufferCount(self.ringSizeBuffer)

        pfResult = self.pfCam.AddStream(self.pfStream)
        if pfResult != pf.Error.NONE:
            ExitWithErrorPrompt("Error setting stream", pfResult)

    def _releaseLastBuffer(self):
        ''' 
        necessary to call 
        if function getLastImage is called with the copyImage=False 
        '''
        #Release frame buffer, otherwise ring buffer will get full
        self.pfStream.ReleaseBuffer(self.pfBuffer)
        self.pfBufferReleased=True
            
    def getLastImage(self,waitForValidImage=True, copyImage=True):
        ''' get image from the buffer 
        waitForValidImage = True ... wait till image is retrieved
                        =  False ... return image can be None '''

        #pfBuffer = 0
        pfImage = pf.PFImage()
        imageData = None
        pfResult = pf.Error.NONE

        if self.pfStream:

            if not self.pfBufferReleased:
                self._releaseLastBuffer()

            [pfResult, self.pfBuffer] = self.pfStream.GetNextBuffer()
            self.pfBufferReleased= False
            #print(pfResult)

            # loop for waiting for a valid image
            while waitForValidImage and pfResult != pf.Error.NONE:
                time.sleep(0.003)
                self._releaseLastBuffer()
                [pfResult, self.pfBuffer] = self.pfStream.GetNextBuffer()

                #print(pfResult)
                #print(waitForValidImage)

            if pfResult == pf.Error.NONE:
                #Get image object from buffer
                self.pfBuffer.GetImage(pfImage)
                imageData = np.array(pfImage, copy = copyImage)

            if copyImage:
                self._releaseLastBuffer()
 
        return (pfResult, imageData)

    def getDarkImage(self,nDark = 1):
        ''' get the dark image, averaged over nDark Images'''
        
        for ii in range(nDark):
        # take image
            _, im = self.getLastImage()
            if ii== 0:
                imAve = im.astype('uint64')
            else:
                imAve += im
        
        self.darkImage = imAve/nDark
        return self.darkImage
        
    def imageDataToSpectralCube(self,imageData,darkImage=None,spectralCorrection=True):
        ''' convert image to hyper spectral cube '''

        if darkImage is not None:
            self.darkImage = darkImage

        # renaming and subtract ing darkImage
        image = imageData  - self.darkImage

        # works only for camera with 25 pixel per super-pixel
        #select only full superpixels
        im2D = np.squeeze(image[0:image.shape[0]//5*5, 0:image.shape[1]//5*5])

        # convert to  spectral cube (x,y, pixels)
        im3D = np.reshape(im2D,(im2D.shape[0],-1,5))
        im4D = np.swapaxes(np.reshape(np.swapaxes(im3D,0,1),(im3D.shape[1],-1,im3D.shape[2],5)),0,1)
        imxyl = np.reshape(im4D,(im4D.shape[0],im4D.shape[1],-1))

        # spectral unmixing correction
        if spectralCorrection:
            imxyw = np.tensordot(imxyl,self.pixelChar['coef_pixels'], axes = (2,0))

        imwxy = np.moveaxis(imxyw,-1,0)

        return imwxy

    def DisplayStreamOfSpectralCube(self):
        ''' display SpectalCube in napari '''

        import napari
        from napari.qt.threading import thread_worker
        import time

        import pyqtgraph as pg
        from PyQt5.QtGui import QColor, QPen

        pxAve = 10

        def calculateSpec():
            ''' calculate the spectra '''
            mySpec.clear()

            # set backgound if bcg point is set
            try:
                mybcg = np.sum(
                        viewer.layers['SpectraCube'].data[
                        :,
                        int(bcg_layer.data[0,0])-pxAve:int(bcg_layer.data[0,0])+pxAve,
                        int(bcg_layer.data[0,1])-pxAve:int(bcg_layer.data[0,1])+pxAve
                        ], axis=(1,2))
            except:
                mybcg = None


            myPoints = points_layer.data
            for ii in np.arange(myPoints.shape[0]):
                try:
                    temp = np.sum(
                        viewer.layers['SpectraCube'].data[
                        :,
                        int(myPoints[ii,0])-pxAve:int(myPoints[ii,0])+pxAve,
                        int(myPoints[ii,1])-pxAve:int(myPoints[ii,1])+pxAve
                        ], axis = (1,2))
                    if mybcg is not None:
                        temp = temp / mybcg *100
                    mySpec.append(temp)
                except:
                    mySpec.append(0*self.pixelChar['wv'])


        def draw_spectraGraph():
            # remove all lines
            p1.clear()
            lineplotList.clear()

            calculateSpec()
            try:
                for ii in np.arange(len(mySpec)):
                    mypen = QPen(QColor.fromRgbF(*list(
                        points_layer.face_color[ii])))
                    lineplot = p1.plot(pen= mypen)
                    lineplot.setData(self.pixelChar['wv'], mySpec[ii])
                    lineplotList.append(lineplot)
            except:
                print('error occured in draw_spectraGraph')

            # check if bcg point is set
            try:
                mybcg = viewer.layers['SpectraCube'].data[
                        :,int(bcg_layer.data[0,0]),int(bcg_layer.data[0,1])
                        ]
                p1.setTitle(f'Transmission')
                p1.setLabel('left', 'percentage', units='a.u.')
            except:
                p1.setTitle(f'Spectra')
                p1.setLabel('left', 'Intensity', units='a.u.')


        def update_spectraGraph():
            ''' update the lines in the spectra graph '''
            nPoints = np.array(points_layer.data.shape[0])
            myPoints = points_layer.data
            try:
                for ii, myline in enumerate(lineplotList):
                    mypen = QPen(QColor.fromRgbF(*list(
                        points_layer.face_color[ii])))
                    myline.setData(self.pixelChar['wv'],mySpec[ii], pen = mypen)

            except:
                print('error occured in update_spectraGraph')


        @thread_worker
        def update_Graph():
            while True:
                calculateSpec()
                update_spectraGraph()
                yield 
                time.sleep(0.1)


        def update_layer(new_image):
            rawlayer.data = new_image
            layer.data = self.imageDataToSpectralCube(new_image)


        @thread_worker
        def yieldHSImage():
            while True:
                (pfResult, imageData) = self.getLastImage(waitForValidImage=False)
                if pfResult == pf.Error.NONE:
                    yield imageData
                    time.sleep(0.1)
                #Display stream statistics
                #pfStreamStats = self.pfStream.GetStreamStatistics()
                #print(pfStreamStats, end="\r")

        worker = yieldHSImage()
        worker.yielded.connect(update_layer)
        worker2 = update_Graph()

        # start napari        
        viewer = napari.Viewer()

        # add pyqt widget 
        p1 = pg.plot()
        lineplotList = []
        mySpec = []
        p1.setAspectLocked(True)
        p1.setTitle(f'Spectra')
        styles = {'color':'r', 'font-size':'20px'}
        p1.setLabel('left', 'Intensity', units='a.u.')
        p1.setLabel('bottom', 'Wavelength ', units= 'nm')
        p1.setXRange(self.pixelChar['wv'][0],self.pixelChar['wv'][-1], padding=0)
        p1.setYRange(0,100)
        viewer.window.add_dock_widget(p1)

        # add camera layer
        # create empty image
        im = np.zeros((self.height,self.width))
        new_image = self.imageDataToSpectralCube(im)
        layer = viewer.add_image(new_image, rgb=False, colormap="gray", 
                                            name='SpectraCube', blending='additive')

        # dark layer
        darklayer = viewer.add_image(self.darkImage, rgb=False, colormap="gray", 
                                            name='Dark', scale = (0.2,0.2) ,blending='additive')

        # raw image
        rawlayer = viewer.add_image(new_image, rgb=False, colormap="gray", 
                                            name='Raw', scale = (0.2,0.2),  blending='additive')


        # add point layer
        points_layer = viewer.add_points(name='points', size=5)
        # connect changes in data in this layer
        points_layer.events.data.connect(draw_spectraGraph)

        # add background layer
        bcg_layer = viewer.add_points(name='bcg_points', size=5)
        # connect changes in data in this layer
        bcg_layer.events.data.connect(draw_spectraGraph)

        worker.start()
        worker2.start()
        napari.run()


    def DisplayStreamOfImages(self):
        ''' display the live images on the screen '''
        import msvcrt
        import cv2 as cv
        from matplotlib import pyplot as plt

        print("Image W:%d H:%d" % (self.width, self.height))

        #Compute display window size
        aspectRatio = float(self.height) / float(self.width)
        displaySize = (1000, int(1000 * aspectRatio))

        imageSize = self.height * self.width

        print("\n\n\nPress any key to stop capturing")
        print("\033[1A\033[1A\033[1A", end="")

        #Loop over stream frames
        while not msvcrt.kbhit():
            
            (pfResult, imageData) = self.getLastImage(waitForValidImage=False)
            
            if pfResult == pf.Error.NONE:

                resizedImage = cv.resize(imageData, displaySize)
                cv.imshow("Captured frame", resizedImage)
                cv.waitKey(1)

                #Show equalized image
                equalized = (resizedImage/(resizedImage.max()- resizedImage.min())*255).astype(np.uint8) 
                - resizedImage.min() 
                cv.imshow("Equalized frame", equalized)
                

            #Display stream statistics
            pfStreamStats = self.pfStream.GetStreamStatistics()
            print(pfStreamStats, end="\r")

    def StartAcquisition(self):
        #Start grabbing images into stream

        pfResult = self.pfCam.Grab()
        if pfResult != pf.Error.NONE:
            ExitWithErrorPrompt("Could not start grab process", pfResult)

    def StopAcquisition(self):
        #Stop frame grabbing
        pfResult = self.pfCam.Freeze()
        if pfResult != pf.Error.NONE:
            ExitWithErrorPrompt("Error stopping grab process", pfResult)

    def DisconnectCamera(self):
        #Disconnect camera
        pfResult = self.pfCam.Disconnect()
        if pfResult != pf.Error.NONE:
            ExitWithErrorPrompt("Error disconnecting", pfResult)

if __name__ == "__main__":

    pfCam = Photonfocus()
    pfCam.PrepareCamera()
    pfCam.SetParameter("PixelFormat", "Mono12")
    pfCam.SetParameter("ExposureTime", 1000)

    pfCam.GetCalibrationData()

    pfCam.StartAcquisition()
    print('Acquiring dark images')
    pfCam.getDarkImage(nDark = 10)
    print('Acquiring dark images finished')
    #pfCam.DisplayStreamOfImages()
    pfCam.DisplayStreamOfSpectralCube()
    pfCam.StopAcquisition()
    pfCam.DisconnectCamera()
    #PromptEnterAndExit()


# %%
