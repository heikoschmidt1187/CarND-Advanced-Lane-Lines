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
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        #maximum number of iterations to average
        self.max_n = 30

    def reset(self):
        __init__()

    def update(self, points_x, points_y):
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

            #print("Best_fit: ", self.best_fit)


            # safe current fit points
            self.allx = points_x
            self.ally = points_y

            # TODO: self.line_base_pos, self.radius_of_curvature

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
