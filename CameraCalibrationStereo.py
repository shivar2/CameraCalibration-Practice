import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob
import os

chessboardSize = (24,17)
imgSize = (460,680)
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

def stereo_get_points(images1, images2):

    total_objPoints = []
    total_imgPoints1 = []
    total_imgPoints2 = []

    objpoints = np.zeros((chessboardSize[0]*chessboardSize[1],3), np.float32)
    objpoints[:, :2] = np.mgrid[:chessboardSize[0], :chessboardSize[1]].T.reshape(-1,2)

    for image1, image2 in zip(images1, images2):
        img1 = cv.imread(image1)
        img2 = cv.imread(image2)

        imgGray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        imgGray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

        ret1, corners1 = cv.findChessboardCorners(imgGray1, chessboardSize, None)
        ret2, corners2 = cv.findChessboardCorners(imgGray2, chessboardSize, None)

        if ret1 and ret2==True:
            total_objPoints.append(objpoints)

            refCorners1 = cv.cornerSubPix(imgGray1, corners1, (11,11), (-1,-1), criteria)
            total_imgPoints1.append(refCorners1)

            refCorners2 = cv.cornerSubPix(imgGray2, corners2, (11,11), (-1,-1), criteria)
            total_imgPoints2.append(refCorners2)

    return total_objPoints, total_imgPoints1, total_imgPoints2

def camera_calibration(objpoints, imgpoints):
    ret, cameraMatrix, dist, rot, trans = cv.calibrateCamera(objpoints, imgpoints, imgSize, None, None)

    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, imgSize, 1, imgSize)

    return cameraMatrix, dist, newCameraMatrix, roi


def stereo_camera_calibration(objpoints, imgpoints1, imgpoints2):
    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC
    

    cameraMatrix1, dist1, newCameraMatrix1, roi1 = camera_calibration(objpoints, imgpoints1)
    cameraMatrix2, dist2, newCameraMatrix2, roi2= camera_calibration(objpoints, imgpoints2)

    retStereo, newCameraMatrix1, dist1, newCameraMatrix2, dist2, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, imgpoints1, imgpoints2 , newCameraMatrix1, dist1, newCameraMatrix2, dist2, imgSize, criteria=criteria, flags=flags)
    
    # rectifyScale= 1
    # rect1, rect2, projMatrix1, projMatrix2, Q, roi1, roi2= cv.stereoRectify(newCameraMatrix1, dist1, newCameraMatrix2, dist2, imgSize, rot, trans, rectifyScale, imgSize)
    
    return cameraMatrix1, dist1, newCameraMatrix1, roi1, cameraMatrix2, dist2, newCameraMatrix2, roi2

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

def test_stereo_cameraCalibration():
    global chessboardSize
    chessboardSize = (9,6)

    images1 = get_images(path='media\camera-calibration\stereo\cam1\*.png')
    images2 = get_images(path='media\camera-calibration\stereo\cam2\*.png')

    total_objPoints, total_imgPoints1, total_imgPoints2 = stereo_get_points(images1, images2)
    cameraMatrix1, dist1, newCameraMatrix1, roi1, cameraMatrix2, dist2, newCameraMatrix2, roi2 = stereo_camera_calibration(total_objPoints, total_imgPoints1, total_imgPoints2)


    # test
    testImg1 = cv.imread('media\\camera-calibration\\stereo\\cam1\\imageL0.png')
    testImgRGB1 = cv.cvtColor(testImg1, cv.COLOR_BGR2RGB)

    distImg1 = undistrot(testImgRGB1, cameraMatrix1, dist1, newCameraMatrix1, roi1, crop=True)
    cv.imwrite('media\\camera-calibration\\stereo\\test\\0-1-distroted.png', distImg1)

    
    testImg2 = cv.imread('media\\camera-calibration\\stereo\\cam2\\imageR0.png')
    testImgRGB2 = cv.cvtColor(testImg2, cv.COLOR_BGR2RGB)

    distImg2 = undistrot(testImgRGB2, cameraMatrix2, dist2, newCameraMatrix2, roi2, crop=True)
    cv.imwrite('media\\camera-calibration\\stereo\\test\\0-2-distroted.png', distImg2)

    plt.figure(figsize=(8,6))

    plt.subplot(221)
    plt.imshow(testImgRGB1)
    plt.title('Input Image 1')
    plt.axis('off')

    plt.subplot(222)
    plt.imshow(distImg1)
    plt.title('Undistorted Image 1')
    plt.axis('off')

    plt.subplot(223)
    plt.imshow(testImgRGB2)
    plt.title('Input Image 2')
    plt.axis('off')

    plt.subplot(224)
    plt.imshow(distImg2)
    plt.title('Undistorted Image 2')
    plt.axis('off')

    plt.show()

def undistrot_folder(folder_path1, folder_path2):
    global chessboardSize
    chessboardSize = (9,6)
    
    images1 = get_images(path=folder_path1 +'/*.png')
    images2 = get_images(path=folder_path2 +'/*.png')

    total_objPoints, total_imgPoints1, total_imgPoints2 = stereo_get_points(images1, images2)
    cameraMatrix1, dist1, newCameraMatrix1, roi1, cameraMatrix2, dist2, newCameraMatrix2, roi2 = stereo_camera_calibration(total_objPoints, total_imgPoints1, total_imgPoints2)


    for image1, image2 in zip(images1, images2):
        
        testImg1 = cv.imread(image1)

        distImg1 = undistrot(testImg1, cameraMatrix1, dist1, newCameraMatrix1, roi1, crop=True)

        filename1 = os.path.basename(image1)
        output_path1 = os.path.join(folder_path1, 'dist',filename1)
        cv.imwrite(output_path1, distImg1)

        
        testImg2 = cv.imread(image2)
        distImg2 = undistrot(testImg2, cameraMatrix2, dist2, newCameraMatrix2, roi2, crop=True)

        filename2 = os.path.basename(image2)
        output_path2 = os.path.join(folder_path2, 'dist',filename2)
        cv.imwrite(output_path2, distImg2)


if __name__ == '__main__':
    # test_stereo_cameraCalibration()
    undistrot_folder(
        'media\\camera-calibration\\stereo\\cam1\\',
        'media\\camera-calibration\\stereo\\cam2\\'
    )
