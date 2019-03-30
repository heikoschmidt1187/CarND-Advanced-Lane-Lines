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
        TODO
        """
        self.__init__()

    def update(self, points_x, points_y, roi_warped_points):
        """
        TODO
        """
        # TODO: handle internal structures, history, maybe sanity, curvature, pos,...

        try:
            # calculate the current fit
            self.current_fit = np.polyfit(points_y, points_x, 2)
            #print("Current fit: ", self.current_fit)

            # calculate the diff between current and best fit
            self.diffs = self.current_fit - self.best_fit
            #print("Best fit: ", self.best_fit)
            #print("Diffs: ", self.diffs)

            # calculate new best fit and save history
            if len(self.recent_fit) == self.max_n:
                self.recent_fit = self.recent_fit[1:]

            self.recent_fit.append(self.current_fit)
            #print("Recent: ", self.recent_fit)

            sum = [np.array([False])]
            for r in self.recent_fit:
                sum = sum + r
            self.best_fit = (sum / len(self.recent_fit))[0]

            # TODO: weighted average!!!

            #print("Best_fit: ", self.best_fit)

            # TODO: this should be fitted points! this can save calculation later in processing class
            # TODO: refactor for better efficiency!!!
            # TODO: restore_last and curvature handling on fallback
            # safe current fit points
            self.ally = np.linspace(0, roi_warped_points[2][1] - 1, roi_warped_points[2][1])
            self.allx = self.get_fit_x(self.ally)

            self.calculate_metrics(roi_warped_points)

            self.detected = True

        except:
            self.detected = False
            self.current_fit = self.best_fit

    def get_fit_x(self, ploty):
        """
        TODO
        """
        # Generate x and y values for plotting
        try:
            fitx = self.best_fit[0]*ploty**2 + self.best_fit[1]*ploty + self.best_fit[2]
        except TypeError:
            # TODO: check if this is needed and sufficient

            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            fitx = 1*ploty**2 + 1*ploty

        return fitx

    def restore_last(self, roi_warped_points, points_x = None, points_y = None):
        """
        TODO
        """

        # when calling with no parameter, we just keep the last state activated
        # and keep signaling a detected line to absorb small flaws of a few
        # frames - otherwise calculate with new input data

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
        self.diffs = self.current_fit - self.best_fit

        # TODO: check for corner cases!

        # on new input points, update lane
        if (points_x is not None) and (points_y is not None):
            self.update(points_x, points_y)
        else:
            self.calculate_metrics(roi_warped_points)

    def calculate_metrics(self, roi_warped_points):
        """
        TODO
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

        # from the roi pixels we calculate the offset from the camera if the car
        # is at middle --> ie. half bottom roi with is 0
        x_center = (roi_warped_points[0][0] + \
            (roi_warped_points[1][0] - roi_warped_points[0][0]) / 2) * xm_per_pix

        top_fitx = self.best_fit[0]*roi_warped_points[0][1]**2 + \
            self.best_fit[1]*roi_warped_points[0][1] + \
            self.best_fit[2]
        base_fitx = self.best_fit[0]*roi_warped_points[2][1]**2 + \
            self.best_fit[1]*roi_warped_points[2][1] + \
            self.best_fit[2]
        # TODO: use shape
        self.line_base_pos = base_fitx * xm_per_pix - x_center

        ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
        # TODO: eliminate calculation

        fit_cr = np.polyfit(self.ally * ym_per_pix, self.allx * xm_per_pix, 2)

        self.radius_of_curvature = ((1 + (2*fit_cr[0]*719*ym_per_pix + \
            fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])


    @staticmethod
    def sanity_check(left_line, right_line):
        """
        TODO
        """
        # check curvatures
        # check horizontal separation distance
        # check lines are roughly parallel
        return True
