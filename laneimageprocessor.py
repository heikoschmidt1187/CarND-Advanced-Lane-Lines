import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from line import Line

# Define a class for processing a lane image
class LaneImageProcessor():

    def __init__(self, camera):
        # flag to show debug images while processing
        self.showDebug = False
        self.noSanityCheck = False
        self.debugFOV = False
        # camera object for undistortion
        self.camera = camera
        # current frame
        self.currentFrame = []
        # current grayscale
        self.currentGray = []
        # current HLS
        self.currentHLS = []
        # current thresholds
        self.abs_sobel_x = []
        self.abs_sobel_y = []
        self.mag_grad = []
        self.dir_grad = []
        self.color_thresh_R = []
        self.color_thresh_S = []
        self.color_thresh_H = []
        self.combined_threshold = []

        # define the world and perspective space points
        # currently they are hard coded, in a real world environment with a fixed
        # mounted camera this may be calculated
        self.world = np.float32(
            [[592, 450],
            [687, 450],
            [1095, 707],
            [210, 707]])

        self.perspective = np.float32(
            [[260, 0],
            [920, 0],
            [920, 720],
            [260, 720]])

        # left and right line
        self.lines = {'left' : Line(self.perspective), 'right' : Line(self.perspective)}
        self.unplausible_lines_ctr = 0
        self.max_unplausible_lines = 10

        # shapes for debug visualization
        self.debug_image_shape = (720, 1920, 3)
        self.debug_image_small_size = (640, 360)

    def reset(self, camera):
        """
        Resets the current state
        """
        self.__init__(camera)
        self.lines['left'].reset(self.perspective)
        self.lines['right'].reset(self.perspective)


    def process(self, frame, showDebugImages=True, reset=False, noSanityCheck=False, debugFOV=False):
        """
        `frame` Input frame in RGB color space to be processed
        `showDebugImages` Input flag to show debug images while processing

        This function processes a lane frame according to the defined image
        processing pipeline.

        returns the current to lane lines
        """
        self.showDebug = showDebugImages
        self.noSanityCheck = noSanityCheck
        self.debugFOV = debugFOV

        # on reset triggered, reset lanes
        if reset == True:
            self.lines['left'].reset(self.perspective)
            self.lines['right'].reset(self.perspective)

        # undistort the frame and save to object state
        self.currentFrame = self.camera.undistort(frame)

        # convert to grayscale and hsl for further processing and smooth to reduce noise
        self.currentGray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        self.currentGray = cv2.GaussianBlur(self.currentGray, (5, 5), 0)

        self.currentHLS = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)
        self.currentHLS = cv2.GaussianBlur(self.currentHLS, (5, 5), 0)

        # check for useful sobel operators
        self.abs_sobel_x = self.abs_sobel_threshold('x', kernel_size=7, threshold=(15, 100))
        self.abs_sobel_y = self.abs_sobel_threshold('y', kernel_size=7, threshold=(15, 100))
        self.mag_grad = self.mag_sobel_threshold(kernel_size=7, threshold=(30, 100))
        self.dir_grad = self.direction_sobel_threshold(kernel_size=31, threshold=(0.5, 1.0))

        # check for useful color operators
        self.color_thresh_S = np.zeros_like(self.currentHLS[:,:,2])
        S = self.currentHLS[:,:,2]
        self.color_thresh_S[(S >= 170) & (S <= 255)] = 1

        # combine the thresholds
        grad_combined = np.zeros_like(self.currentGray)
        self.combined_threshold = np.zeros_like(self.currentGray)

        self.combined_threshold[
            ((self.abs_sobel_x == 1) & (self.abs_sobel_y == 1))
            | ((self.mag_grad == 1) & (self.dir_grad == 1))
            | (self.color_thresh_S == 1)
            ] = 1


        # get the bird's eye view of the combined threshold image
        if self.debugFOV == True:
            FOV = self.perspective_transform('b', self.currentFrame)
        else:
            birds_view_thresh = self.perspective_transform('b', self.combined_threshold)

            # detect lanes in the current frame
            lane_detection, lanes_valid = self.detect_lanes(birds_view_thresh)

        #self.show_debug_plots()
        return self.visualize_lanes(lane_detection, lanes_valid)


    def visualize_lanes(self, debug_viz, lanes_valid):
        """
        `debug_viz` Input debug image from detect_lanes operation to be included in the debug view
        `lanes_valid` Input global flag if the lanes are valid

        This function takes the current frame, a debug image and the current lanes to calculate
        the output visualization for the lane detection algorithm.

        Returns an annotated imaged with detected lanes overlayed
        """

        # draw overlay image for current frame
        overlay = np.zeros_like(self.currentFrame)
        frame = np.copy(self.currentFrame)

        # lines are only drawn if they have valid data
        if lanes_valid == True:

            # Recast the x and y points into usable format for cv2.fillPoly()
            pts_left = np.array([np.transpose(np.vstack([self.lines['left'].allx, self.lines['left'].ally]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([self.lines['right'].allx, self.lines['right'].ally])))])
            pts = np.hstack((pts_left, pts_right))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(overlay, np.int_([pts]), (0,255, 0))

            # re-warp lane_detection and overlap with current frame
            warp = self.perspective_transform('r', overlay)
            frame = cv2.addWeighted(frame, 1, warp, 0.3, 0.)

            # annotate image with curavtures and bases
            rad_left = self.lines['left'].radius_of_curvature
            rad_right = self.lines['left'].radius_of_curvature
            curvature_text = 'Radius of curvature: ' + str(round((rad_left + rad_right) / 2, 2)) + \
                'm (left: ' + str(round(rad_left, 2)) + 'm, right: ' + str(round(rad_right, 2)) + 'm)'
            line_base_text = 'Base left: ' + str(round(self.lines['left'].line_base_pos, 2)) + \
                'm Base right: ' + str(round(self.lines['right'].line_base_pos, 2)) + 'm'

            deviation_of_center = abs(self.lines['left'].line_base_pos) - abs(self.lines['right'].line_base_pos)
            pos_text = 'Vehicle is ' + str(abs(round(deviation_of_center, 2))) + 'm ' + \
                ('left ' if (deviation_of_center < 0) else 'right ') + 'of center'

            cv2.putText(frame, curvature_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, line_base_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, pos_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


        if self.showDebug == True:
            # compose a return image consisting of analysis steps
            debug_image = np.zeros(self.debug_image_shape, dtype=np.uint8)
            debug_image[0:frame.shape[0], 0:frame.shape[1]] = frame
            debug_image[0:self.debug_image_small_size[1], frame.shape[1]:] = \
                cv2.resize(debug_viz, self.debug_image_small_size)

            combined = np.dstack((self.combined_threshold, self.combined_threshold,
                self.combined_threshold)) * 255
            debug_image[self.debug_image_small_size[1]:frame.shape[0], frame.shape[1]:] = \
                cv2.resize(combined, (640, 360))

            return debug_image
        else:
            # if debug isn't enabled, return the annotated frame only
            return frame


    def search_around_poly(self, bird_view, line):
        """
        `bird_view` Input bird view image of threshold to analyse for lanes
        `line` Input line with current fit to use

        This function finds lane pixels based on a margin around a prior detected
        poly line. This avoids setting up sliding windows in each cycle and
        saves calculation time.

        returns the X and Y values for left and right side where the detected
        lane pixels are located
        """
        # margin around already detected poly lines
        margin = 50

        # get activated pixels
        nonzero = bird_view.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # build the search poly around the current fit
        lane_inds = ((nonzerox > (line.current_fit[0]*(nonzeroy**2) + line.current_fit[1]*nonzeroy +
                    line.current_fit[2] - margin)) & (nonzerox < (line.current_fit[0]*(nonzeroy**2) +
                    line.current_fit[1]*nonzeroy + line.current_fit[2] + margin)))

        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        out_img = np.dstack((bird_view, bird_view, bird_view)) * 255
        window_img = np.zeros_like(out_img)

        line_window1 = np.array([np.transpose(np.vstack([line.allx-margin, line.ally]))])
        line_window2 = np.array([np.flipud(np.transpose(np.vstack([line.allx+margin,
                              line.ally])))])
        line_pts = np.hstack((line_window1, line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([line_pts]), (0,255, 0))

        return x, y, window_img


    def find_lane_pixels(self, bird_view):
        """
        `bird_view` Input bird view image of threshold to analyse for lanes

        This function finds lane pixels based on a histogram approach. Therefor
        it doesn't use any prior information, but instead starts without any
        previous knowledge.

        returns the X and Y values for left and right side where the detected
        lane pixels are located
        """
        # get histogram peaks as a starting point in the lower image half
        lower_half = bird_view[bird_view.shape[0] // 2:, :]
        histogram = np.sum(lower_half, axis=0)

        # split histogram into left and right, as car should be always between
        # the lines (except lange change) and camera is mounted in center
        out_img = np.dstack((bird_view, bird_view, bird_view)) * 255

        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # sliding window parameters
        number_of_windows = 8 # total number of windows vertically
        margin = 100 # pixel margin for each window left and right of center
        minpix = 50 # minimun number of pixels to detect for center replacing

        # window height depends on the number of sliding windows
        window_height = np.int(bird_view.shape[0] // number_of_windows)

        # get position of all pixel in binary image that are one
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
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, out_img

    def detect_lanes(self, bird_view):
        """
        `bird_view` Input bird view image of threshold to analyse for lanes

        This function detects lane lines in a given bird's eye view binary threshold
        image.

        returns a RGB channel image based on the input threshold image, composed
        with the analysis results
        """
        # detect the X and Y positions of all relevant lane pixels - depending on
        # history of detected lanes
        if (self.lines['left'].detected == True) and (self.lines['right'].detected == True):
            leftx, lefty, overlay1 = self.search_around_poly(bird_view, self.lines['left'])
            rightx, righty, overlay2 = self.search_around_poly(bird_view, self.lines['right'])
            out_img = np.dstack((bird_view, bird_view, bird_view)) * 255
            out_img = cv2.addWeighted(out_img, 1, overlay1, 0.3, 0)
            out_img = cv2.addWeighted(out_img, 1, overlay2, 0.3, 0)
        else:
            leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(bird_view)

        # fit left and right
        self.lines['left'].update(leftx, lefty)
        self.lines['right'].update(rightx, righty)

        # sanity check for lines
        restore_valid = True

        if (self.noSanityCheck == True) or \
            (Line.sanity_check(self.lines['left'], self.lines['right']) == True):
            # sanity check passed, reset counter for consecutive unplausible lines
            self.unplausible_lines_ctr = 0
        else:
            # sanity check failed, increment counter
            self.unplausible_lines_ctr = self.unplausible_lines_ctr + 1

            if self.unplausible_lines_ctr > self.max_unplausible_lines:
                # if too many consecutive unplausible lines occured, we start fresh
                # by using the sliding window approach based on historgram
                #print("Max number of unplausible lines reached, start new")
                leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(bird_view)
                restore_valid = restore_valid and \
                    self.lines['left'].restore_last(leftx, lefty)
                restore_valid = restore_valid and \
                    self.lines['right'].restore_last(rightx, righty)

            else:
                # restore the last valid lines if there
                #print("Unplausible lines, keep last pair")
                restore_valid = restore_valid and self.lines['left'].restore_last()
                restore_valid = restore_valid and self.lines['right'].restore_last()

        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        cv2.polylines(img=out_img,
                      pts=np.int32(np.dstack((self.lines['left'].allx, self.lines['left'].ally))),
                      isClosed=False,
                      color=(255, 255, 0),
                      thickness=10,
                      lineType=cv2.LINE_8)
        cv2.polylines(img=out_img,
                      pts=np.int32(np.dstack((self.lines['right'].allx, self.lines['right'].ally))),
                      isClosed=False,
                      color=(255, 255, 0),
                      thickness=10,
                      lineType=cv2.LINE_8)

        return out_img, restore_valid

    def perspective_transform(self, direction, srcImage):
        """
        `direction` Input that defines the direction to transform (b - world to bird's view, else back)
        `srcImage` Input that will be transformed

        This function does a perspective transform in a given direction on a given source image. It
        relies on four defined points in the world space and four points in the perspective space

        Returns the warped image
        """

        # if debug view for FOV is activated, draw the ROI on the original image and transform it
        if self.debugFOV == True:
            srcImage = cv2.line(srcImage, (self.world[0][0], self.world[0][1]),
                (self.world[1][0], self.world[1][1]), (255, 0, 0), 1)
            srcImage = cv2.line(srcImage, (self.world[1][0], self.world[1][1]),
                (self.world[2][0], self.world[2][1]), (255, 0, 0), 1)
            srcImage = cv2.line(srcImage, (self.world[2][0], self.world[2][1]),
                (self.world[3][0], self.world[3][1]), (255, 0, 0), 1)
            srcImage = cv2.line(srcImage, (self.world[3][0], self.world[3][1]),
                (self.world[0][0], self.world[0][1]), (255, 0, 0), 1)

        # get perspective perspective transform
        if direction == 'b':
            # do a bird view transform
            M = cv2.getPerspectiveTransform(self.world, self.perspective)
        else:
            M = cv2.getPerspectiveTransform(self.perspective, self.world)

        # transform the image based on the source and destination points
        transformed = cv2.warpPerspective(srcImage, M, (srcImage.shape[1],
            srcImage.shape[0]), flags=cv2.INTER_LINEAR)

        # if debug view for FOV is activated, draw the ROI on the original image and transform it
        if self.debugFOV == True:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24,9))
            ax1.imshow(srcImage, cmap='gray')
            ax2.imshow(transformed, cmap='gray')
            plt.show()

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
            f, ((orig, gray, s), (sx, sy, mag), (dir, st, combined)) = \
                plt.subplots(3, 3, figsize=(24,9))

            orig.imshow(self.currentFrame)
            orig.set_title("Original frame")
            gray.imshow(self.currentGray, cmap='gray')
            gray.set_title("Grayscale")
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

            st.imshow(self.color_thresh_S, cmap='gray')
            st.set_title("S-Channel color threshold")
            combined.imshow(self.combined_threshold, cmap='gray')
            combined.set_title("Combined Threshold")

            plt.show()
