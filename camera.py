import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define a class to handle all camera specific calculations
class Camera():
    def __init__(self):
        # flag if the camera has been calibrated yet
        self.calibrated = False
        # the camera distortion coefficients vector
        self.distCoeffs = []
        # the camera intrinsics matrix
        self.mtx = []

    def calibrate(self, calibImagePathsList, showDebugImages=False, nx=9, ny=6):
        """
        `calibImagePathsList` Input vector of paths to calibration files
        `showDebugImages` Input to show debug images while calibration if True
        `nx` Input number of corners in X direction
        `ny` Input number of corners in Y direction

        Calibrates the camera based on the given calibration filename list and
        the number of chessboard corners in x and y direction

        returns if the calibration has been successful
        """

        print("Camera Calibrating...")

        # start uncalibrated when running calibration routine
        self.calibrated = False

        if calibImagePathsList and (nx > 0) and (ny > 0):
            # build vectors for imagepoints and objectpoints
            imgpoints = [] # 2D points in image plane
            objpoints = [] # 3D points in real world

            # prepare the object points according to the parameters (z stays 0
            # as the chessboards are on a flat surface)
            objp = np.zeros((nx * ny, 3), np.float32)
            objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) # x, y coordinates

            if showDebugImages == True:
                cv2.namedWindow("Calibration Debug", cv2.WINDOW_AUTOSIZE)
                cv2.moveWindow("Calibration Debug", 0, 0)

            for path in calibImagePathsList:

                # load the image
                image = mpimg.imread(path)

                # TODO: check if load was successful
                # convert image to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                # find chessboard corners for further detection
                ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

                if ret == True:
                    imgpoints.append(corners)
                    objpoints.append(objp)

                    # draw the corners on debug
                    if showDebugImages == True:
                        cv2.drawChessboardCorners(image, (nx, ny), corners, ret)
                        cv2.imshow("Calibration Debug", image)
                        cv2.waitKey(1000)

                # calibrate the camera if valid points have been found
                if len(objpoints) > 0:
                    ret, self.mtx, self.distCoeffs, rvecs, tvecs = \
                        cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], \
                         None, None)

                    self.calibrated = True

            if showDebugImages == True:
                cv2.destroyWindow("Calibration Debug")

        return self.calibrated

    def undistort(self, image):
        """
        `image` Input image to be undistorted

        This operation undistorts images according to the calibration values

        returns undistorted image
        """

        # make a copy of the image
        retImage = np.copy(image)

        """
        The image can only be undistorted if the disotrion coefficients and
        the intrinsics matrix have been calculated -- meaning the camera is
        calibrated
        """
        if self.calibrated == True:
            retImage = cv2.undistort(image, self.mtx, self.distCoeffs, None, self.mtx)
        else:
            print("Warning: Camera not calibrated, no undistortion possible")

        return retImage
