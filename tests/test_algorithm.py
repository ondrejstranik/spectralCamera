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




