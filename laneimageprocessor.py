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
        # current thresholds
        self.abs_sobel_x = []
        self.abs_sobel_y = []
        self.mag_grad = []
        self.dir_grad = []
        self.color_thresh_R = []
        self.color_thresh_S = []
        self.color_thresh_H = []
        self.combined_threshold = []

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
        self.abs_sobel_x = self.abs_sobel_threshold('x', threshold=(35, 150))
        self.abs_sobel_y = self.abs_sobel_threshold('y', threshold=(35, 150))
        self.mag_grad = self.mag_sobel_threshold(threshold=(50, 150))
        self.dir_grad = self.direction_sobel_threshold(threshold=(0.7, 1.3))

        # check for useful color operators
        self.color_thresh_R = np.zeros_like(self.currentFrame[:,:,0])
        R = self.currentFrame[:,:,0]
        self.color_thresh_R[(R >= 220) & (R <= 255)] = 1

        self.color_thresh_S = np.zeros_like(self.currentHLS[:,:,2])
        S = self.currentHLS[:,:,2]
        self.color_thresh_S[(S >= 100) & (S <= 255)] = 1

        self.color_thresh_H = np.zeros_like(self.currentHLS[:,:,0])
        H = self.currentHLS[:,:,0]
        self.color_thresh_H[(H >= 20) & (H <= 100)] = 1

        self.combined_threshold = np.zeros_like(self.currentGray)
        self.combined_threshold[ \
            ((self.color_thresh_R == 1) & (self.color_thresh_S == 1)) | \
            ((self.abs_sobel_x == 1) & (self.mag_grad == 0))] = 1

        # bird view handling
        birds_view_thresh = self.perspective_transform('b', self.combined_threshold)



        self.show_debug_plots()

    def perspective_transform(self, direction, srcImage):
        """
        TODO
        """
        # define the world and perspective space points
        world = np.float32(
            [[603, 443],
            [677, 443],
            [1048, 670],
            [260, 670]])

        # TODO: use image shape
        perspective = np.float32(
            [[260, 0],
            [1020, 0],
            [1020, 720],
            [260, 720]])

        """
        srcImage = cv2.line(srcImage, (603, 443), (677, 443), (255, 0, 0), 1)
        srcImage = cv2.line(srcImage, (677, 443), (1048, 670), (255, 0, 0), 1)
        srcImage = cv2.line(srcImage, (1048, 670), (260, 670), (255, 0, 0), 1)
        srcImage = cv2.line(srcImage, (260, 670), (603, 443), (255, 0, 0), 1)
        """

        # get perspective perspective transform
        if direction == 'b':
            # do a bird view transform
            M = cv2.getPerspectiveTransform(world, perspective)
        else:
            M = cv2.getPerspectiveTransform(perspective, world)

        transformed = cv2.warpPerspective(srcImage, M, (1280, 720), flags=cv2.INTER_LINEAR)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,9))
        ax1.imshow(srcImage, cmap='gray')
        ax2.imshow(transformed, cmap='gray')

        return transformed

    def direction_sobel_threshold(self, kernel_size=3, threshold=(0, np.pi / 2)):
        """
        `kernel_size` Input for kernel size of sobel operator
        `threshold` Input tuple for lower and upper threshold in rad

        This function calculates the gradients and thresholds the direction based
        on given angles

        returns a binary image based on the given thresholds
        """

        # calculate the sobel
        sobelx = cv2.Sobel(self.currentGray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobely = cv2.Sobel(self.currentGray, cv2.CV_64F, 0, 1, ksize=kernel_size)

        # calculate the gradient direction
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

        # calculate the binary output with respect to thresholds
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= threshold[0]) & (absgraddir <= threshold[1])] = 1

        # Return the binary image
        return binary_output


    def mag_sobel_threshold(self, kernel_size=3, threshold=(0, 255)):
        """
        `kernel_size` Input for kernel size of sobel operator
        `threshold` Input tuple for lower and upper threshold

        This function calculates the magnitude of the gradients detected by the
        sobel operator in X and Y direction.

        returns a binary image based on the given thresholds
        """

        # calculate the sobel
        sobelx = cv2.Sobel(self.currentGray, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobely = cv2.Sobel(self.currentGray, cv2.CV_64F, 0, 1, ksize=kernel_size)

        # calculate the gradient magnitude
        magnitude = np.sqrt(sobelx**2 + sobely**2)

        # rescale to 8 bit
        scale = np.max(magnitude)/255
        magnitude = (magnitude / scale).astype(np.uint8)

        # calculate the binary output with respect to thresholds
        binary_output = np.zeros_like(magnitude)
        binary_output[(magnitude >= threshold[0]) & (magnitude <= threshold[1])] = 1
        return binary_output


    def abs_sobel_threshold(self, orientation='x', kernel_size=3, threshold=(0, 255)):
        """
        `orientation` Input for setting the sobel operator gradient orientation (x, y)
        `kernel_size` Input for kernel size of sobel operator
        `threshold` Input tuple for lower and upper threshold

        This function calculates a binary image mask according to the absolute
        sobel operation on a given gradient, based on a lower and upper
        threshold.

        returns a binary image
        """
        # calculate the sobel depending on the orientation
        if orientation == 'x':
            abs_sobel = np.absolute(cv2.Sobel(self.currentGray, cv2.CV_64F, 1, 0, \
                ksize=kernel_size))
        elif orientation == 'y':
            abs_sobel = np.absolute(cv2.Sobel(self.currentGray, cv2.CV_64F, 0, 1, \
                ksize=kernel_size))
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
            f, ((orig, r, g, b), (gray, h, l, s), (sx, sy, mag, dir), (rt, st, rh, combined)) = \
            plt.subplots(4, 4, figsize=(24,9))

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

            sx.imshow(self.abs_sobel_x, cmap='gray')
            sx.set_title("Sobel X Threshold")
            sy.imshow(self.abs_sobel_y, cmap='gray')
            sy.set_title("Sobel Y Threshold")
            mag.imshow(self.mag_grad, cmap='gray')
            mag.set_title("Magnitude Threshold")
            dir.imshow(self.dir_grad, cmap='gray')
            dir.set_title("Direction Threshold")

            rt.imshow(self.color_thresh_R, cmap='gray')
            rt.set_title("R-Channel color threshold")
            st.imshow(self.color_thresh_S, cmap='gray')
            st.set_title("S-Channel color threshold")
            rh.imshow(self.color_thresh_H, cmap='gray')
            rh.set_title("H-Channel color threshold")
            combined.imshow(self.combined_threshold, cmap='gray')
            combined.set_title("Combined Threshold")

            plt.show()
