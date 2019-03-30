import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = [np.array([False])]
        #polinomial coefficients for the last n fits of the lane
        self.recent_fit = []
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = 0
        #distance in meters of vehicle center from the line
        self.line_base_pos = 0
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        #maximum number of iterations to average
        self.max_n = 25

    def reset(self):
        """
        This function resets the current Line back into it's original state defined
        in the class' constructor
        """
        self.__init__()

    def update(self, points_x, points_y, roi_warped_points):
        """
        `points_x`  Input of all X coordinates of found relevant pixels that
                    may define a lane
        `points_y`  Input of all Y coordiantes of found relevant pixels that
                    may define a lane
        `roi_warped_points` Input that consists of four points in the bird's
                            view image space

        This function updates the internal state of the line object. It therefor
        takes the pixel coordinates in X and Y direction, fits a second order
        polynom and tracks the history over the last defined number of frames
        """

        try:
            # calculate the current fit
            self.current_fit = np.polyfit(points_y, points_x, 2)

            # calculate the diff between current and best fit
            self.diffs = self.current_fit - self.best_fit

            # calculate new best fit and save history
            if len(self.recent_fit) == self.max_n:
                self.recent_fit = self.recent_fit[1:]
            self.recent_fit.append(self.current_fit)

            # TODO: weighted average!!!
            sum = [np.array([False])]
            for r in self.recent_fit:
                sum = sum + r
            self.best_fit = (sum / len(self.recent_fit))[0]

            # TODO: refactor for better efficiency!!!
            # TODO: restore_last and curvature handling on fallback
            # TODO: refactor -- linspace only needs to be called once in the whole run
            # safe current fit points
            self.ally = np.linspace(0, roi_warped_points[2][1] - 1, roi_warped_points[2][1])
            self.allx = self.get_fit_x(self.ally)

            # calculate the real world unit metrics
            self.calculate_metrics(roi_warped_points)

            # set the lane detected
            self.detected = True

        except:
            # something went wrong, set detected to false and switch to current
            # best fit
            self.detected = False
            self.current_fit = self.best_fit

    def get_fit_x(self, ploty):
        """
        `ploty` Input of the Y linspace to calculate the X coordinates

        Calculates the X coordinates for the current best fit of the line.

        Returns the X values corresponding to the ploty input space
        """
        # Generate x and y values for plotting
        try:
            fitx = self.best_fit[0]*ploty**2 + self.best_fit[1]*ploty + self.best_fit[2]
        except TypeError:
            # Avoids an error if best_fit is still none or incorrect
            print('The function failed to fit a line!')
            fitx = 1*ploty**2 + 1*ploty

        return fitx

    def restore_last(self, roi_warped_points, points_x = None, points_y = None):
        """
        `roi_warped_points` Input that consists of four points in the bird's
                            view image space
        `points_x`  Input of all X coordinates of found relevant pixels that
                    may define a lane
        `points_y`  Input of all Y coordiantes of found relevant pixels that
                    may define a lane

        This function restores the last fit and discards the current one. If additionally
        the pixel points are given, instead of restoring the last fit, it will
        completely initialize the line and start from scratch.

        Returns True if the restore or finding of new line was successful, else False
        """

        # when calling with no parameter, we just keep the last state activated
        # and keep signaling a detected line to absorb small flaws of a few
        # frames - otherwise calculate with new input data

        # on new input points, reset and update lane
        if (points_x is not None) and (points_y is not None):
            self.reset()
            self.update(points_x, points_y, roi_warped_points)

            return self.detected

        elif len(self.recent_fit) >= 2:
            # ensure detected
            self.detected = True

            # remove current broken fit from recent fits
            self.recent_fit = self.recent_fit[:-1]

            # make last valid recent fit to current fit
            self.current_fit = self.recent_fit[-1]

            # calculate new best fit
            sum = [np.array([False])]
            for r in self.recent_fit:
                sum = sum + r
            self.best_fit = (sum / len(self.recent_fit))[0]
            # TODO: weighted average!!!

            # re calculate diffs
            self.riffs = self.current_fit - self.best_fit

            # we need to re-calculate the metrics
            self.calculate_metrics(roi_warped_points)

            return True
        else:
            # if not, there's currently no way out
            return False


    def calculate_metrics(self, roi_warped_points):
        """
        `roi_warped_points` Input that consists of four points in the bird's
                            view image space

        This function translate the current line information into real world
        units. This is done for the radius of the curvature and the base position
        with to the mount position of the camera on the car.
        """
        # line base pos is calculated through the roi information
        # the used four point ROI has two points at the bottom that are straight
        # with respect to the bottom - as this points are right next to the lines,
        # they can be translated from pixels into meters with the knowledge of
        # a U.S. highway standard lane - this is an apprximation, but should be
        # good enough for this project
        # U.S. regulations minimum lane width: 3.7m
        xm_per_pix = 3.7 / (roi_warped_points[1][0] - roi_warped_points[0][0])

        # each dashed line is 3m long --> about 30m for warped image
        ym_per_pix = 35 / (roi_warped_points[2][1] - roi_warped_points[0][1])

        # TODO: put metric distances to ROI definition if possible
        # TODO: make global to use in other functions

        # from the roi pixels we calculate the offset from the camera if the car
        # is at middle --> ie. half bottom roi with is 0
        x_center = (roi_warped_points[0][0] + \
            (roi_warped_points[1][0] - roi_warped_points[0][0]) / 2) * xm_per_pix

        # calculate the base point (near the car) with respect to the ROI
        base_fitx = self.best_fit[0]*roi_warped_points[2][1]**2 + \
            self.best_fit[1]*roi_warped_points[2][1] + \
            self.best_fit[2]

        # calculate the base point for the line in m according to the camera
        # as origin point
        self.line_base_pos = base_fitx * xm_per_pix - x_center

        # calculate the radius of the curvature by fitting a polynom through
        # the current X and Y points in real world units
        fit_cr = np.polyfit(self.ally * ym_per_pix, self.allx * xm_per_pix, 2)

        self.radius_of_curvature = ((1 + (2*fit_cr[0]*719*ym_per_pix + \
            fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])


    @staticmethod
    def sanity_check(left_line, right_line, roi_warped_points):
        """
        `left_line` Input of the left line object to sanitize
        `right_line` Input of the right line object to sanitize
        `roi_warped_points` Input that consists of four points in the bird's
                            view image space

        This function does a sanity check on two given lines according to their
        radius of curvature, base distance and parallelity.

        Returns True if the sanity check passed, else False
        """

        # TODO: maybe use confidence values for all three criteria

        # check horizontal separation distance
        if abs(right_line.line_base_pos - left_line.line_base_pos) > 4.0:
            print("Line base positions too far from each other")
            return False

        # check lines are roughly parallel
        # if base pos and raduius of both lines are ok, it should be enough
        # to check the X distances of a few points with respect to their y positions
        # so slice the Y points into chunks and check
        chunksize = 200
        length = min(len(left_line.ally), len(right_line.ally))
        xm_per_pix = 3.7 / (roi_warped_points[1][0] - roi_warped_points[0][0])
        # TODO: global definition

        bias = None
        for i in range(0, length, chunksize):

            # take x at car as bias
            if bias is None:
                bias = abs(right_line.allx[i] - left_line.allx[i]) * xm_per_pix
            else:
                if abs(bias - abs(right_line.allx[i] - left_line.allx[i])*xm_per_pix) > 1.0:
                    print("Lines are not parallel")
                    return False

        # check curvatures -- the curvatures for left and right should be roughly
        # in the same magitude -- check for error
        if abs(left_line.radius_of_curvature - right_line.radius_of_curvature) > 200:
            print("Line radius of curvature too different")
            return False

        return True
