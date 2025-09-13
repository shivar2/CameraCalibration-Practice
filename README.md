# CameraCalibration-Practice

This repository provides a workflow and examples for calibrating cameras. It includes a Jupyter notebook that lets you run each calibration step individually and inspect the results, as well as additional scripts/notebooks for stereo and monocular calibration.

# Contents
- CameraCalibration-Practic.ipynb — Interactive notebook to execute each calibration step separately and inspect results
- Supporting scripts for:
    - Stereo camera calibration
    - Monocular camera calibration
 
# Quick Start
1. Prerequisites

- Python (recommended: 3.8+)
- OpenCV (cv2)
- NumPy
- Matplotlib (or your preferred plotting library)

  
2. Dataset

- The notebook is designed to work with chessboard pattern images.
- I used images from the following repository:
https://github.com/niconielsen32/ComputerVision/tree/master

3. Running the notebook

- Open CameraCalibration-Practic.ipynb
- Run the cells step by step:
  - **Step 1: Find object points and image points**
    - Detect chessboard corners in each image
    - Prepare the 3D object points (e.g., on the chessboard plane) and corresponding 2D image points
   
  - **Step 2: Estimate the camera calibration matrix**
    - Compute the camera intrinsic matrix K and distortion coefficients
    - Optionally estimate rotation and translation vectors for each image
   
  - **Step 3: Estimate and remove image distortion**
    - Use the distortion model to undistort images
    - Validate calibration by reprojecting points and checking reprojection error
    
- Visualize results as instructed in the notebook (plots of reprojected points, undistorted images, etc.)


# Key Concepts
- **Object points**: <br/> The 3D coordinates of the chessboard corners in the calibration pattern’s coordinate system (typically with Z = 0 for a flat board).
  
- **Image points**: <br/> The 2D coordinates of the detected corners in the corresponding images.
  
- **Camera calibration matrix**: <br/> The intrinsic matrix *K* that maps 3D camera coordinates to image coordinates.
  
- **Distortion parameters**: <br/> Coefficients that model lens distortion (radial and tangential) which are removed during undistortion.
  
  
Mathematically, the standard calibration solves for the parameters that minimize the reprojection error between observed image points and the projected object points using the camera model.


# Additional Calibrations
- **Stereo calibration**:  
  A separate workflow for calibrating a stereo pair, estimating both intrinsics and the extrinsic relationship (rotation and translation between cameras), as well as rectification maps.
  
- **Monocular calibration**:  
  A streamlined workflow focusing on a single camera, useful when only one sensor is available or when stereo calibration is not needed.
