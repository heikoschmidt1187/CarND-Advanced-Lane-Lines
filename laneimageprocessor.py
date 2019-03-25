import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from line import Line

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
        # left and right line
        self.lines = {'left' : Line(), 'right' : Line()}

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

        # return image combined of processing images
        debug_image = np.zeros((720, 1920, 3), dtype=np.uint8)
        debug_image[0:720, 0:1280] = self.currentFrame
        detection = np.dstack((birds_view_thresh, birds_view_thresh, birds_view_thresh)) * 255
        #debug_image[0:360, 1280:] = cv2.resize(bird, (640, 360))

        detection = self.detect_lanes(birds_view_thresh)
        debug_image[0:360, 1280:] = cv2.resize(detection, (640, 360))

        return debug_image

        #self.show_debug_plots()

    def find_lane_pixels(self, bird_view):
        """
        TODO
        """
        # get histogram peaks as a starting point in the lower image half
        lower_half = bird_view[bird_view.shape[0] // 2:, :]
        histogram = np.sum(lower_half, axis=0)

        # TODO: visualize histogram in image

        # split histogram into left and right, as car should be always between
        # the lines and camera is mounted in center (except lange chane!)
        out_img = np.dstack((bird_view, bird_view, bird_view)) * 255

        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # sliding window parameters
        number_of_windows = 9 # total number of windows vertically
        margin = 100 # pixel margin for each window left and right of center
        minpix = 50 # minimun number of pixels to detect for center replacing

        # window height depends on the number of sliding windows
        window_height = np.int(bird_view.shape[0] // number_of_windows)

        # get position of all pixel in binary image that are on
        nonzero = bird_view.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # set current mid to histogram based values first
        leftx_current = leftx_base
        rightx_current = rightx_base

        # left and right pixel indizes
        left_lane_inds = []
        right_lane_inds = []

        for window in range(number_of_windows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = bird_view.shape[0] - (window+1)*window_height
            win_y_high = bird_view.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2)

            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

         # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img

    def detect_lanes(self, bird_view):
        """
        TODO
        """

        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(bird_view)

        # Fit a second order polynomial to each using `np.polyfit`
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, bird_view.shape[0]-1, bird_view.shape[0] )
        try:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1*ploty**2 + 1*ploty
            right_fitx = 1*ploty**2 + 1*ploty

        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # Plots the left and right polynomials on the lane lines
        """
        TODO
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        """

        return out_img

    def perspective_transform(self, direction, srcImage):
        """
        TODO
        """
        # define the world and perspective space points
        world = np.float32(
            [[603, 443],
            [677, 443],
            [1095, 707],
            [210, 707]])

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

        """
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,9))
        ax1.imshow(srcImage, cmap='gray')
        ax2.imshow(transformed, cmap='gray')
        plt.show()
        """

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
