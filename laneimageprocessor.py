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
        self.noSanityCheck = False
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
        self.unplausible_lines_ctr = 0
        self.max_unplausible_lines = 10

        # define the world and perspective space points
        """
        self.world = np.float32(
            [[603, 443],
            [677, 443],
            [1095, 707],
            [210, 707]])
        """
        self.world = np.float32(
            [[576, 460],
            [703, 460],
            [1095, 707],
            [210, 707]])

        # TODO: use image shape
        """
        self.perspective = np.float32(
            [[260, 0],
            [920, 0],
            [920, 720],
            [260, 720]])
        """
        self.perspective = np.float32(
            [[260, 0],
            [920, 0],
            [920, 720],
            [260, 720]])

    def process(self, frame, showDebugImages=True, reset=False, noSanityCheck=False):
        """
        `frame` Input frame in RGB color space to be processed
        `showDebugImages` Input flag to show debug images while processing

        This function processes a lane frame according to the defined image
        processing pipeline.

        returns the current to lane lines
        """
        self.showDebug = showDebugImages
        self.noSanityCheck = noSanityCheck

        if reset == True:
            self.lines['left'].reset()
            self.lines['right'].reset()

        # convert to grayscale and hsl for further processing
        self.currentFrame = frame

        # smooth to reduce noise
        self.currentGray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        self.currentGray = cv2.GaussianBlur(self.currentGray, (5, 5), 0)

        self.currentHLS = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)
        self.currentHLS = cv2.GaussianBlur(self.currentHLS, (5, 5), 0)

        # check for useful sobel operators
        """
        self.abs_sobel_x = self.abs_sobel_threshold('x', threshold=(25, 255))
        self.abs_sobel_y = self.abs_sobel_threshold('y', threshold=(25, 255))
        self.mag_grad = self.mag_sobel_threshold(threshold=(34, 255))
        self.dir_grad = self.direction_sobel_threshold(threshold=(0.7, 1.3))
        """
        self.abs_sobel_x = self.abs_sobel_threshold('x', kernel_size=7, threshold=(15, 100))
        self.abs_sobel_y = self.abs_sobel_threshold('y', kernel_size=7, threshold=(15, 100))
        self.mag_grad = self.mag_sobel_threshold(kernel_size=7, threshold=(30, 100))
        self.dir_grad = self.direction_sobel_threshold(kernel_size=31, threshold=(0.5, 1.0))

        # check for useful color operators
        self.color_thresh_R = np.zeros_like(self.currentFrame[:,:,0])
        R = self.currentFrame[:,:,0]
        self.color_thresh_R[(R >= 220) & (R <= 255)] = 1

        self.color_thresh_S = np.zeros_like(self.currentHLS[:,:,2])
        S = self.currentHLS[:,:,2]
        """
        self.color_thresh_S[(S >= 80) & (S <= 255)] = 1
        """
        self.color_thresh_S[(S >= 170) & (S <= 255)] = 1

        self.color_thresh_H = np.zeros_like(self.currentHLS[:,:,0])
        H = self.currentHLS[:,:,0]
        self.color_thresh_H[(H >= 17) & (H <= 195)] = 1

        # combine first the gradient filters and add to satuarion threshold mask
        grad_combined = np.zeros_like(self.currentGray)
        self.combined_threshold = np.zeros_like(self.currentGray)

        """
        self.combined_threshold[ \
            ((self.color_thresh_R == 1) & (self.color_thresh_S == 1)) | \
            ((self.abs_sobel_x == 1) & (self.abs_sobel_y == 0))] = 1
        self.combined_threshold[(self.color_thresh_R == 1) | (self.color_thresh_S == 1)] = 1
        grad_combined[((self.abs_sobel_x == 1) & (self.abs_sobel_y == 1)) \
            | ((self.mag_grad == 1) & (self.dir_grad == 1))] == 1
        self.combined_threshold[(self.color_thresh_S == 1) | (grad_combined == 1)] = 1
        """
        """
        self.combined_threshold[(self.abs_sobel_x == 1 )
            | ((self.mag_grad == 1) & (self.dir_grad == 1)) | (self.color_thresh_S == 1)] = 1
        """

        """
        self.combined_threshold[((self.abs_sobel_x == 1) & (self.abs_sobel_y == 1) & (self.color_thresh_H == 1))
        | (self.color_thresh_S == 1)] = 1
        """
        self.combined_threshold[
            ((self.abs_sobel_x == 1) & (self.abs_sobel_y == 1))
            | ((self.mag_grad == 1) & (self.dir_grad == 1))
            | (self.color_thresh_S == 1)
            ] = 1

        # get the bird's eye view of the combined threshold image
        #b = self.perspective_transform('b', self.currentFrame)
        birds_view_thresh = self.perspective_transform('b', self.combined_threshold)

        # detect lanes in the current frame
        lane_detection, lanes_valid = self.detect_lanes(birds_view_thresh)

        #self.show_debug_plots()
        return self.visualize_lanes(lane_detection, lanes_valid)


    def visualize_lanes(self, debug_viz, lanes_valid):
        """
        TODO
        """

        # draw overlay image for current frame
        overlay = np.zeros_like(self.currentFrame)
        frame = np.copy(self.currentFrame)

        if lanes_valid == True:
            ploty = np.linspace(0, frame.shape[0]-1, frame.shape[0] )
            left_fit_x = self.lines['left'].get_fit_x(ploty)
            right_fit_x = self.lines['right'].get_fit_x(ploty)

            # Recast the x and y points into usable format for cv2.fillPoly()
            pts_left = np.array([np.transpose(np.vstack([left_fit_x, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, ploty])))])
            pts = np.hstack((pts_left, pts_right))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(overlay, np.int_([pts]), (0,255, 0))

            # re-warp lane_detection and overlap with current frame
            warp = self.perspective_transform('r', overlay)
            frame = cv2.addWeighted(frame, 1, warp, 0.3, 0.)

            # annotate image with curavtures and bases
            # TODO: mean curvature and car pos from center
            rad_left = self.lines['left'].radius_of_curvature
            rad_right = self.lines['left'].radius_of_curvature
            curvature_text = 'Radius of curvature: ' + str(round((rad_left + rad_right) / 2, 2)) + \
                'm (left: ' + str(round(rad_left, 2)) + 'm, right: ' + str(round(rad_right, 2)) + 'm)'
            line_base_text = 'Base left: ' + str(round(self.lines['left'].line_base_pos, 2)) + \
                'm Base right: ' + str(round(self.lines['right'].line_base_pos, 2)) + 'm'

            deviation_of_center = abs(self.lines['left'].line_base_pos) - abs(self.lines['right'].line_base_pos)
            pos_text = 'Vehicle is ' + str(abs(round(deviation_of_center, 2))) + 'm ' + ('left ' if (deviation_of_center < 0) else 'right ') + 'of center'

            cv2.putText(frame, curvature_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, line_base_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, pos_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


        if self.showDebug == True:
            # compose a return image consisting of analysis steps
            # TODO: use shapes instead of magic numbers
            debug_image = np.zeros((720, 1920, 3), dtype=np.uint8)
            debug_image[0:720, 0:1280] = frame
            debug_image[0:360, 1280:] = cv2.resize(debug_viz, (640, 360))

            combined = np.dstack((self.combined_threshold, self.combined_threshold,
                self.combined_threshold)) * 255
            debug_image[360:720, 1280:] = cv2.resize(combined, (640, 360))
            return debug_image
        else:
            return frame


    def search_around_poly(self, bird_view, fit):
        """
        `bird_view` Input bird view image of threshold to analyse for lanes
        `fit` Input line fit poly around search needs to be done

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

        lane_inds = ((nonzerox > (fit[0]*(nonzeroy**2) + fit[1]*nonzeroy +
                    fit[2] - margin)) & (nonzerox < (fit[0]*(nonzeroy**2) +
                    fit[1]*nonzeroy + fit[2] + margin)))

        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        ploty = np.linspace(0, bird_view.shape[1], bird_view.shape[0] )
        fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]

        out_img = np.dstack((bird_view, bird_view, bird_view)) * 255
        window_img = np.zeros_like(out_img)

        line_window1 = np.array([np.transpose(np.vstack([fitx-margin, ploty]))])
        line_window2 = np.array([np.flipud(np.transpose(np.vstack([fitx+margin,
                              ploty])))])
        line_pts = np.hstack((line_window1, line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([line_pts]), (0,255, 0))
        #out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

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

        # TODO: histogram visualization

        # split histogram into left and right, as car should be always between
        # the lines (except lange change) and camera is mounted in center
        out_img = np.dstack((bird_view, bird_view, bird_view)) * 255

        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # sliding window parameters
        number_of_windows = 10 # total number of windows vertically
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
        # TODO: depending on history don't always start a fresh search
        if (self.lines['left'].detected == True) and (self.lines['right'].detected == True):
            leftx, lefty, overlay1 = self.search_around_poly(bird_view, self.lines['left'].current_fit)
            rightx, righty, overlay2 = self.search_around_poly(bird_view, self.lines['right'].current_fit)
            out_img = np.dstack((bird_view, bird_view, bird_view)) * 255
            out_img = cv2.addWeighted(out_img, 1, overlay1, 0.3, 0)
            out_img = cv2.addWeighted(out_img, 1, overlay2, 0.3, 0)
        else:
            leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(bird_view)

        # fit left and right
        self.lines['left'].update(leftx, lefty, self.perspective)
        self.lines['right'].update(rightx, righty, self.perspective)

        # TODO: use lane_valid instead of restore_valid and secure if update did not find anything!

        # sanity check for lines
        restore_valid = True

        if (self.noSanityCheck == True) or \
            (Line.sanity_check(self.lines['left'], self.lines['right'], self.perspective) == True):
            # sanity check passed, reset counter for consecutive unplausible lines
            self.unplausible_lines_ctr = 0
        else:
            # sanity check failed, increment counter
            self.unplausible_lines_ctr = self.unplausible_lines_ctr + 1

            if self.unplausible_lines_ctr > self.max_unplausible_lines:
                # if too many consecutive unplausible lines occured, we start fresh
                # by using the sliding window approach based on historgram
                print("Max number of unplausible lines reached, start new")
                leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(bird_view)
                restore_valid = restore_valid and \
                    self.lines['left'].restore_last(self.perspective, leftx, lefty)
                restore_valid = restore_valid and \
                    self.lines['right'].restore_last(self.perspective, rightx, righty)

            else:
                # restore the last valid lines if there
                print("Unplausible lines, keep last pair")
                restore_valid = restore_valid and \
                    self.lines['left'].restore_last(self.perspective)
                restore_valid = restore_valid and \
                    self.lines['right'].restore_last(self.perspective)

        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        return out_img, restore_valid

    def perspective_transform(self, direction, srcImage):
        """
        TODO
        """
        # TODO: globals to avoid magic numbers in update and restore call

        """
        srcImage = cv2.line(srcImage, (self.world[0][0], self.world[0][1]),
            (self.world[1][0], self.world[1][1]), (255, 0, 0), 1)
        srcImage = cv2.line(srcImage, (self.world[1][0], self.world[1][1]),
            (self.world[2][0], self.world[2][1]), (255, 0, 0), 1)
        srcImage = cv2.line(srcImage, (self.world[2][0], self.world[2][1]),
            (self.world[3][0], self.world[3][1]), (255, 0, 0), 1)
        srcImage = cv2.line(srcImage, (self.world[3][0], self.world[3][1]),
            (self.world[0][0], self.world[0][1]), (255, 0, 0), 1)
        """

        # get perspective perspective transform
        if direction == 'b':
            # do a bird view transform
            M = cv2.getPerspectiveTransform(self.world, self.perspective)
        else:
            M = cv2.getPerspectiveTransform(self.perspective, self.world)

        # TODO: image shape
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
