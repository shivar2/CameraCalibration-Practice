import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob

chessboardSize = (24,17)
imgSize = (1440,1080)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def set_imgSize(imgName):
    global imgSize
    img = cv.imread(imgName)
    imgSize = (img.shape[1], img.shape[0])

def get_images(path):
    images = glob.glob(path)

    if images[0]:
        set_imgSize(images[0])

    return images

def get_points(images):

    total_objPoints = []
    total_imgPoints = []

    objpoints = np.zeros((chessboardSize[0]*chessboardSize[1],3), np.float32)
    objpoints[:, :2] = np.mgrid[:chessboardSize[0], :chessboardSize[1]].T.reshape(-1,2)

    for image in images:
        img = cv.imread(image)
        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(imgGray, chessboardSize, None)

        if ret==True:
            total_objPoints.append(objpoints)
            refCorners = cv.cornerSubPix(imgGray, corners, (11,11), (-1,-1), criteria)
            total_imgPoints.append(refCorners)

    return total_objPoints, total_imgPoints

def camera_calibration(objpoints, imgpoints):
    ret, cameraMatrix, dist, rot, trans = cv.calibrateCamera(objpoints, imgpoints, imgSize, None, None)

    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, imgSize, 1, imgSize)

    return cameraMatrix, dist, newCameraMatrix, roi

def crop_img(img, roi):
    x, y, w, h = roi
    img = img[y:y+h, x:x+w]
    return img

def undistrot(img, cameraMatrix, dist, newCameraMatrix, roi, crop=True):
    mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, imgSize, 5)
    distMapImg = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    if crop:
        distMapImg = crop_img(distMapImg, roi)

    return distMapImg

def test_monocular_cameraCalibration():

    images = get_images(path='media\camera-calibration\single-camera\*.png')

    total_objPoints, total_imgPoints = get_points(images)

    cameraMatrix, dist, newCameraMatrix, roi = camera_calibration(total_objPoints, total_imgPoints)

    testImg = cv.imread('media\\camera-calibration\\single-camera\\test-images\\15.png')
    testImgRGB = cv.cvtColor(testImg, cv.COLOR_BGR2RGB)

    distImg = undistrot(testImgRGB, cameraMatrix, dist, newCameraMatrix, roi, crop=True)
    cv.imwrite('media\\camera-calibration\\single-camera\\test-images\\15-distroted.png', distImg)

    plt.figure(figsize=(8,6))

    plt.subplot(121)
    plt.imshow(testImgRGB)
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(distImg)
    plt.title('Undistorted Image')
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    test_monocular_cameraCalibration()