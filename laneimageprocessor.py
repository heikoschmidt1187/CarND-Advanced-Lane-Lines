import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define a class for processing a lane image
class LaneImageProcessor():

    def __init__(self):
        # flag to show debug images while processing
        self.showDebug = False
        # current frame
        currentFrame = []
        # current grayscale
        currentGray = []
        # current HLS
        currentHLS = []

    def process(self, frame, showDebugImages=False):
        """
        `frame` Input frame in RGB color space to be processed
        `showDebugImages` Input flag to show debug images while processing

        This function processes a lane frame according to the defined image
        processing pipeline.

        returns the current to lane lines
        """
        self.showDebug = showDebugImages

        # convert to grayscale and hsl for further processing
        self.currentFrame = frame
        self.currentGray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        self.currentHLS = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)

        # check for useful sobel operators
        sobelx = self.abs_sobel_threshold('x', (35, 150))
        sobely = self.abs_sobel_threshold('y', (35, 150))

        lower = 35
        upper = 150

        while True:
            sobelx = self.abs_sobel_threshold('x', (lower, upper))
            sobely = self.abs_sobel_threshold('y', (lower, upper))

            cv2.imshow("Gray", self.currentGray)
            cv2.imshow("sobelX", sobelx * 255)
            cv2.imshow("sobelY", sobely * 255)


            key = cv2.waitKey()

            if key == 27:
                break
            elif 'q' == chr(key & 255):
                lower = (lower + 1) % 255
                print('Lower: ', lower)
            elif 'a' == chr(key & 255):
                lower = (lower - 1)
                if lower < 0:
                    lower = 0
                print('Lower: ', lower)
            elif 'w' == chr(key & 255):
                upper = (upper + 1) % 255
                print('Upper: ', upper)
            elif 's' == chr(key & 255):
                upper = (upper - 1)
                if upper < 0:
                    upper = 0
                print('Upper: ', upper)



        self.show_debug_plots()

    def abs_sobel_threshold(self, orientation='x', threshold=(0, 255)):
        """
        `orientation` Input for setting the sobel operator gradient orientation (x, y)
        `threshold` Input touble for lower and upper threshold

        This function calculates a binary image mask according to the absolute
        sobel operation on a given gradient, based on a lower and upper
        threshold.

        returns a binary image
        """
        # calculate the sobel depending on the orientation
        if orientation == 'x':
            abs_sobel = np.absolute(cv2.Sobel(self.currentGray, cv2.CV_64F, 1, 0))
        elif orientation == 'y':
            abs_sobel = np.absolute(cv2.Sobel(self.currentGray, cv2.CV_64F, 0, 1))
        else:
            abs_sobel = np.zeros_like(self.currentGray)
            print("None")

        # rescale the sobel to uint8 type
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        # calculate the binary output with respect to thresholds
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= threshold[0]) & (scaled_sobel <= threshold[1])] = 1

        return binary_output

    def show_debug_plots(self):

        # Show debug images if selected - this may change during development
        if self.showDebug == True:
            f, ((orig, r, g, b), (gray, h, l, s)) = plt.subplots(2, 4, figsize=(24,9))
            orig.imshow(self.currentFrame)
            orig.set_title("Original frame")
            r.imshow(self.currentFrame[:,:,0], cmap='gray')
            r.set_title("RGB R-Channel")
            g.imshow(self.currentFrame[:,:,1], cmap='gray')
            g.set_title("RGB G-Channel")
            b.imshow(self.currentFrame[:,:,2], cmap='gray')
            b.set_title("RGB B-Channel")
            gray.imshow(self.currentGray, cmap='gray')
            gray.set_title("Grayscale")
            h.imshow(self.currentHLS[:,:,0], cmap='gray')
            h.set_title("HLS H-Channel")
            l.imshow(self.currentHLS[:,:,1], cmap='gray')
            l.set_title("HLS L-Channel")
            s.imshow(self.currentHLS[:,:,2], cmap='gray')
            s.set_title("HLS S-Channel")
            plt.show()
