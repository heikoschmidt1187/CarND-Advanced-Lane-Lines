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
    imageProcessor = LaneImageProcessor(camera)

    # calibrate the Camera
    images = glob.glob('camera_cal/calibration*.jpg')
    #calibrated = camera.calibrate(images, debugMode)
    calibrated = camera.calibrate(images)

    if calibrated == False:
        print("Camera calibration not successful")
        sys.exit()

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

    # start by using a static test image to implement pipeline
    testimage = mpimg.imread("test_images/straight_lines1.jpg")
    #testimage = mpimg.imread("test_images/test2.jpg")
    testimages = glob.glob('test_images/*.jpg')

    # undistort one of the given testimages
    if debugMode == True:
        test2 = mpimg.imread('test_images/test2.jpg')
        test2undist = camera.undistort(test2)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        ax1.imshow(test2)
        ax1.set_title("Distorted image")
        ax2.imshow(test2undist)
        ax2.set_title("Undistorted image")
        plt.show()


    """
    # sample images
    for curImage in testimages:
        print(curImage)
        testimage = mpimg.imread(curImage)
        debug_image = imageProcessor.process(testimage, debugMode, True, True)

        plt.imshow(debug_image)
        plt.show()

    """

    # imageProcessor.process(testimage, debugMode, True, debugFOV=True)

    imageProcessor.reset(camera)
    test_output1 = 'output_videos/project_video_output.mp4'
    clip1 = VideoFileClip('project_video.mp4')
    test_clip1 = clip1.fl_image(imageProcessor.process)
    test_clip1.write_videofile(test_output1, audio=False)

    imageProcessor.reset(camera)
    test_output2 = 'output_videos/challenge_video_output.mp4'
    clip2 = VideoFileClip('challenge_video.mp4')
    test_clip2 = clip2.fl_image(imageProcessor.process)
    test_clip2.write_videofile(test_output2, audio=False)

    imageProcessor.reset(camera)
    test_output3 = 'output_videos/harder_challenge_video_output.mp4'
    clip3 = VideoFileClip('harder_challenge_video.mp4')
    test_clip3 = clip3.fl_image(imageProcessor.process)
    test_clip3.write_videofile(test_output3, audio=False)
