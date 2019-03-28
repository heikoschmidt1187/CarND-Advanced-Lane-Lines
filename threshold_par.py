import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

def abs_sobel_threshold(img, orientation='x', kernel_size=3, threshold=(0, 255)):
    """
    `orientation` Input for setting the sobel operator gradient orientation (x, y)
    `kernel_size` Input for kernel size of sobel operator
    `threshold` Input tuple for lower and upper threshold

    This function calculates a binary image mask according to the absolute
    sobel operation on a given gradient, based on a lower and upper
    threshold.

    returns a binary image
    """
    gray = cv2.GaussianBlur(img, (5, 5), 0)

    # calculate the sobel depending on the orientation
    if orientation == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, \
            ksize=kernel_size))
    elif orientation == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, \
            ksize=kernel_size))
    else:
        abs_sobel = np.zeros_like(gray)
        print("None")

    # rescale the sobel to uint8 type
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # calculate the binary output with respect to thresholds
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= threshold[0]) & (scaled_sobel <= threshold[1])] = 1

    return binary_output

def direction_sobel_threshold(img, kernel_size=3, threshold=(0, np.pi / 2)):
    """
    `kernel_size` Input for kernel size of sobel operator
    `threshold` Input tuple for lower and upper threshold in rad

    This function calculates the gradients and thresholds the direction based
    on given angles

    returns a binary image based on the given thresholds
    """
    gray = cv2.GaussianBlur(img, (5, 5), 0)

    # calculate the sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

    # calculate the gradient direction
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    # calculate the binary output with respect to thresholds
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= threshold[0]) & (absgraddir <= threshold[1])] = 1

    # Return the binary image
    return binary_output

def mag_sobel_threshold(img, kernel_size=3, threshold=(0, 255)):
    """
    `kernel_size` Input for kernel size of sobel operator
    `threshold` Input tuple for lower and upper threshold

    This function calculates the magnitude of the gradients detected by the
    sobel operator in X and Y direction.

    returns a binary image based on the given thresholds
    """
    gray = cv2.GaussianBlur(img, (5, 5), 0)

    # calculate the sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

    # calculate the gradient magnitude
    magnitude = np.sqrt(sobelx**2 + sobely**2)

    # rescale to 8 bit
    scale = np.max(magnitude)/255
    magnitude = (magnitude / scale).astype(np.uint8)

    # calculate the binary output with respect to thresholds
    binary_output = np.zeros_like(magnitude)
    binary_output[(magnitude >= threshold[0]) & (magnitude <= threshold[1])] = 1
    return binary_output

def nothing(x):
    pass

cv2.namedWindow('image')
"""
cv2.createTrackbar('Low', 'image', 0, 255, nothing)
cv2.createTrackbar('High', 'image', 0, 255, nothing)
"""
cv2.createTrackbar('Low', 'image', 0, 255, nothing)
cv2.createTrackbar('High', 'image', 0, 255, nothing)

testimages = glob.glob('test_images/*.jpg')

for curImage in testimages:

    print(curImage)

    img = cv2.imread(curImage)
    img = cv2.pyrDown(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:,:,1]

    debug_image = np.zeros((360, 640 * 2, 3), dtype=np.uint8)
    debug_image[0:img.shape[0], 0:img.shape[1]] = img

    while(1):

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        low = cv2.getTrackbarPos('Low', 'image')
        high = cv2.getTrackbarPos('High', 'image')

        #binary = abs_sobel_threshold(gray, 'y', kernel_size=3, threshold=(low, high))
        #binary = mag_sobel_threshold(gray, kernel_size=3, threshold=(low, high))
        binary = np.zeros_like(hls)
        binary[(hls > low) & (hls < high)] = 1
        bin = np.dstack((binary, binary, binary)) * 255
        debug_image[0:bin.shape[0], img.shape[1]:] = bin

        cv2.imshow('window', debug_image)


cv2.destroyAllWindows()
