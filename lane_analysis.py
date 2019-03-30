import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from moviepy.editor import VideoFileClip
from line import Line
from camera import Camera
from laneimageprocessor import LaneImageProcessor

debugMode = True

if __name__ == "__main__":

    # create camera object to handle the dash cam
    camera = Camera()

    # create lane image processor object
    imageProcessor = LaneImageProcessor()

    # calibrate the Camera
    images = glob.glob('camera_cal/calibration*.jpg')
    #calibrated = camera.calibrate(images, debugMode)
    calibrated = camera.calibrate(images)

    if calibrated == False:
        print("Camera calibration not successful")
        sys.exit()

    """
    if debugMode == True:
        # test calibration by showing doing an undistortion
        distortedImage = mpimg.imread(images[0])
        undistortedImage = camera.undistort(distortedImage)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        ax1.imshow(distortedImage)
        ax1.set_title("Distorted image")
        ax2.imshow(undistortedImage)
        ax2.set_title("Undistorted image")
        plt.show()
    """

    # start by using a static test image to implement pipeline
    testimage = mpimg.imread("test_images/straight_lines1.jpg")
    #testimage = mpimg.imread("test_images/test2.jpg")
    testimages = glob.glob('test_images/*.jpg')

    """
    for curImage in testimages:
        print(curImage)
        testimage = mpimg.imread(curImage)
        testimage = camera.undistort(testimage)
        debug_image = imageProcessor.process(testimage, debugMode, True)

        plt.imshow(debug_image)
        plt.show()
    """

    """
    testimage = camera.undistort(testimage)
    imageProcessor.process(testimage, debugMode, True)
    """

    test_output = 'project_video_output.mp4'
    clip1 = VideoFileClip('project_video.mp4')
    test_clip = clip1.fl_image(imageProcessor.process)
    test_clip.write_videofile(test_output, audio=False)

    """
    test_output = 'challenge_video_output.mp4'
    clip1 = VideoFileClip('challenge_video.mp4')
    test_clip = clip1.fl_image(imageProcessor.process)
    test_clip.write_videofile(test_output, audio=False)
    """

    """
    test_output = 'harder_challenge_video_output.mp4'
    clip1 = VideoFileClip('harder_challenge_video.mp4')
    test_clip = clip1.fl_image(imageProcessor.process)
    test_clip.write_videofile(test_output, audio=False)
    """
