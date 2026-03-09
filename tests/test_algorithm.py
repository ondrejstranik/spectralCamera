''' algorithm test '''

import pytest

@pytest.mark.GUI
def test_basisVectors():
    ''' visual check if the basis vector are found '''
    import matplotlib.pyplot as plt
    from spectralCamera.algorithm.basisVectors import lattice_basis_vectors
    import numpy as np

    v1 = 2
    v2 = 1.4
    ang = 30*np.pi/180

    rotmat = np.array([
        [np.cos(ang), np.sin(ang)],
        [-np.sin(ang), np.cos(ang)]
        ])

    true = np.array([
        np.dot(rotmat, np.array([0,v2])),
        np.dot(rotmat, np.array([0,-v2])),
        np.dot(rotmat, np.array([v1,0])),
        np.dot(rotmat, np.array([-v1,0]))
        
        ])

    scale = 3
    n = 100

    com = np.reshape(np.array([[np.dot(rotmat, np.array([(i1+(np.random.rand(1)-0.5)/scale)*v1, (i2+(np.random.rand(1)-0.5)/scale)*v2])) for i2 in range(n)] for i1 in range(n)]),(n*n,2))

    plt.figure()
    plt.scatter(*com.T)


    v1 = lattice_basis_vectors(com)

    plt.figure()
    plt.scatter(*lattice_basis_vectors(com, 10).T, c = 'green')
    plt.scatter(*lattice_basis_vectors(com, 1000).T, c = 'red')
    plt.scatter(*true.T, c = 'blue')
    np.median(v1[0])

@pytest.mark.GUI
def test_fileSIVideo():
    ''' check if video of spectral images are loaded'''

    from spectralCamera.algorithm.fileSIVideo import FileSIVideo
    from spectralCamera.gui.spectralViewer.xywViewer import XYWViewer


    # adjust to your data folder
    folder = r'D:\LPI\24-9-16 pfcamera\video'
    print(f'data folder = {folder}')

    fV = FileSIVideo(folder=folder)
    allImage,wavelength,time = fV.loadAllImage()

    print(f'time in ns = {time}')
    viewer = XYWViewer(allImage,wavelength)
    viewer.run()



@pytest.mark.GUI
def test_gridSuperPixel():
    ''' check if the class gridSuperPixel identify the grid and get the spectral blocks'''

    import numpy as np
    import skimage as ski
    import napari
    from spectralCamera.algorithm.gridSuperPixel import GridSuperPixel

    #example image
    myImage = ski.data.coins()

    #generate grid (not ideal)
    xx,yy = np.meshgrid(np.arange(20),np.arange(10))
    xx = xx + 0.1*np.random.rand(*xx.shape)
    yy = yy + 0.1*np.random.rand(*yy.shape)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)

    position = xx[:,None]*np.array([7.1,20]) + yy[:,None]*np.array([20.1,7.1])
    position = position[np.random.permutation(position.shape[0]),:]

    # characterize the grid
    gridSP = GridSuperPixel()
    gridSP.setGridPosition(position)
    gridSP.getGridInfo()
    gridSP.getPixelIndex()
    gridSP.getPositionOutsideImage(myImage)
    spBlock = gridSP.getSpectraBlock(myImage)
    alignedIm = gridSP.getAlignedImage(spBlock)

    # prepare a sub selected  points
    pointsSelect = (gridSP.imIdx[:,0]%1 == 0 ) & (gridSP.imIdx[:,1]%1 == 0 ) & gridSP.inside
    points = gridSP.position[pointsSelect,:]

    features = {'pointIndex0': gridSP.imIdx[pointsSelect,0],
                'pointIndex1': gridSP.imIdx[pointsSelect,1]
                }
    text = {'string': '[{pointIndex0},{pointIndex1}]',
            'translation': np.array([-30, 0])
            }
    # prepare pointImage 
    pointImage = np.zeros_like(myImage).astype(bool)
    pInt = gridSP.getPositionInt()
    pointImage[pInt[gridSP.inside,0],pInt[gridSP.inside,1]] = True

    # display the images
    viewer = napari.Viewer()
    viewer.add_image(myImage)
    viewer.add_image(pointImage, opacity=0.5)
    viewer.add_points(points,features=features,text=text, size= 50, opacity=0.5)

    # display cut image
    viewer2 = napari.Viewer()
    viewer2.add_image(alignedIm)
    napari.run()
