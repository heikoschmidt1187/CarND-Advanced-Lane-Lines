# Writeup
## Udacity Self Driving Car Engineer - Project Advanced Lane Finding
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[imageUndistort]: ./output_images/calibration/undistort.jpg "Undistorted"
[realUndistImage]: ./output_images/undistortTest2.jpg "Undistorted test2.jpg example image"
[threshSteps]: ./output_images/ThresholdsExample.jpg "Threshold steps"
[combinedThreshold]: ./output_images/combinedThresholdExample.jpg "Combined threshold example"
[warpBird]: ./output_images/WarpBird.jpg "Bird's eye view perspective transform"
[slidingWindow]: ./output_images/SlidingWindow.jpg "Sliding Window Example"
[polynomialExample]: ./output_images/PolynomialExample.jpg "Fitted polynomial example"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### General information on the project solution
For this Udacity project, I decided not to use Jupyter Notebook, but instead create a standalone python implementation. It consists of three python files (camera.py, laneimageprocessor.py and line.py) that contain the functional implementation and one python file (lane_analysis.py) for triggering the functional modules. This is also the entrypoint when running this project (call: # python lane_analysis.py)

This architecture allows to use the functional implementation in different systems - e.g. it's easy to use it in a Jupyter Notebook by just creating class objects and trigger them.

While for solving the problems in this project I used an objectoriented approach, it need's to be said that the implementation isn't fully objectoriented to it's full extend.

In the following parts I will directly link to the files and provide code snippets where needed. Please also consider the comments in the code for a better understanding.

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

I provided the writeup in this Github README.md file you are currently reading, for the sake of simplicity.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

As with all camera related operations, the calibration routine is located in the file camera.py. The function calibrate (line 16 through 81) does a calibration based on a list of paths to calibration images that were taken with the camera. These images have been provided by Udacity in the camera_cal subfolder.

The camera object is created and calibrated in the lane_analysis.py file:

```python
  # create camera object to handle the dash cam
  camera = Camera()

  [...]

  # calibrate the Camera
  images = glob.glob('camera_cal/calibration*.jpg')
  #calibrated = camera.calibrate(images, debugMode)
  calibrated = camera.calibrate(images)

  if calibrated == False:
      print("Camera calibration not successful")
      sys.exit()
```

First in the implementation, I prepared  the so called object points. These are the x, y and z coordinated on the chessboard corners in the real world. As from the pictures I can assume that the images are on a flat plane, all the z components will be 0:

```python
  # prepare the object points according to the parameters (z stays 0
  # as the chessboards are on a flat surface)
  objp = np.zeros((nx * ny, 3), np.float32)
  objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) # x, y coordinates
```

After that, I iterate through all the calibration images and try to find the chessboard corners by using the OpenCV function `cv2.findChessboardCorners`. Befor this, I convert the images to grayscale:

```python
  # load the image
  image = mpimg.imread(path)

  # convert image to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

  # find chessboard corners for further detection
  ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
```

For each successful detected corners, I add the previously generated `objp` to the array of objectpoints and the detected image points to the array `imgpoints`. If enabled, all of the steps are shown in dedicated images.

The image points `imgpoints` are the x and y pixel positions of the chessboard cordners in the image plane.

With `objpoints` and `imgpoints` I then use the function `cv2.calibrateCamera` to obtain the distortion coefficients `distCoeffs` and the camera intrinsics matrix `mtx` that are saved in the internal state of the camera object.

```python
  ret, self.mtx, self.distCoeffs, rvecs, tvecs = \
      cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], \
       None, None)
```

With this, the calibration is done. From this point on it's possible to use the `undistort` function of the camera object to fix distortion for any image taken with the camera. The implementation can be found in the camera.py file lines 82 through 104.

The function takes an image, uses the OpenCV function `cv2.undistort` function with the camera intrinsics matrix and the distortion coefficients, and returns the image. One example looks like this:

```python
  distortedImage = mpimg.imread(images[0])
  undistortedImage = camera.undistort(distortedImage)
```
![Undistortion Image][imageUndistort]

### Pipeline (single images)

The analysis pipeline is implemented in the `LaneImageProcessor` class. A LaneImageProcessor object has to be created first, giving a calibrated camera object:

```python
  # create lane image processor object
  imageProcessor = LaneImageProcessor(camera)
```

One pipeline run has then to be triggered with a call of the `process` function:

```python
  testimage = mpimg.imread(curImage)
  testimage = camera.undistort(testimage)
  debug_image = imageProcessor.process(testimage, debugMode, True, True)
```

The process operation returns an annotated images after the pipeline did run. Details on this will follow.

#### 1. Provide an example of a distortion-corrected image.

As a first step in the pipeline, each frame has to be undistorted to prepare for further analysis. The backround for the undistortion is discribed above. One example for an undistorted real world image is this:

![Real undistorted image][realUndistImage]

In the image pipeline, this is done in file laneimageprocessor.py line 87:

```python
  # undistort the frame and save to object state
  self.currentFrame = self.camera.undistort(frame)
```

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The creation of the binary image is done in laneimageprocessor.py from line 90 to 115. This is done through the usage of several helper functions to create the gradient thresholds.

First, the undistorted frame is converted to grayscale and HLS colorspace for further usage by applying the OpenCV `cv2.cvtColor` method. Additionally, a Gaussian Blur with a kernel size of 5 is applied to reduce noise in the images:

```python
  # convert to grayscale and hsl for further processing and smooth to reduce noise
  self.currentGray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
  self.currentGray = cv2.GaussianBlur(self.currentGray, (5, 5), 0)

  self.currentHLS = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)
  self.currentHLS = cv2.GaussianBlur(self.currentHLS, (5, 5), 0)
```

In the next steps, I applied threshold functions for absolute sobel thresholds in x and y direction, a threshold function for the magnitude of the sobel gradient and a threshold for the direction of the sobel gradient:

```python
  # check for useful sobel operators
  self.abs_sobel_x = self.abs_sobel_threshold('x', kernel_size=7, threshold=(15, 100))
  self.abs_sobel_y = self.abs_sobel_threshold('y', kernel_size=7, threshold=(15, 100))
  self.mag_grad = self.mag_sobel_threshold(kernel_size=7, threshold=(30, 100))
  self.dir_grad = self.direction_sobel_threshold(kernel_size=31, threshold=(0.5, 1.0))
```

You can find the function implementation in laneimageprocessor.py lines 425 to 509.

The parameters for kernel size, lower and upper threshold have been determined using a try and error approach on all of the given example images. I have put a special focus on the ones with lighter road surfaces and the one where the shadows of the trees in the middle of the lane are shining on the road

While the gradient threshold do a good job on dark pavement and elimination horizontal line pixel on the road surface, it fails detecting yellow lines on light road surface. To address this issue, I applied a color threshold on the saturation channel of the HLS image:

```python
  # check for useful color operators
  self.color_thresh_S = np.zeros_like(self.currentHLS[:,:,2])
  S = self.currentHLS[:,:,2]
  self.color_thresh_S[(S >= 170) & (S <= 255)] = 1
```

Again, the thresholds have been determined using try and arror. The following image shows each step on an example image:

![Single Threshold Steps][threshSteps]

Each of the thresholds are represented in a binary image. Next I combined them to get a final binary image to use for further analysis. From the different combinations I tried, the following gave the best results on the testimages and the project video:

```python
  self.combined_threshold[
      ((self.abs_sobel_x == 1) & (self.abs_sobel_y == 1))
      | ((self.mag_grad == 1) & (self.dir_grad == 1))
      | (self.color_thresh_S == 1)
      ] = 1
```

The following image shows the thresholding pipeline on one of the testimages:

![Example combined image][combinedThreshold]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transformation is done with the function `perspective_transform` in laneimageprocessor.py:


```python
  # get the bird's eye view of the combined threshold image
  [...]
  birds_view_thresh = self.perspective_transform('b', self.combined_threshold)
```

You can find the implementation from line 384 to 424. The function takes a direction to transform (from world to bird's eye view or back) and a source image. `perspective_transform` relies on two vertex arrays, defined in the constructor of the LaneImageProcessor class:

Rand perspective space points

```python
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
```

I determined the values by zooming into an image with straight lines from the test_images folder, zooming into the image and determining a quadliteral section to be transformed. These points are saved into the self.world variable of the class. After that, I tested different target coordinated for the bird's eye view. The transformed image should have the same shape like the source frame and I wanted the lines to be straight vertical in the middle of the image, so with curvatures I get enough pixels to analyze without running out of the frame. This led to the values in the variable self.perspective.

In the `perspective_transform` function, first with `cv2.getPerspectiveTransform` a transformation matrix is determined based on the above point array, and then a perspective warping with `cv2.waprPerspective` is done with this matrix.

With the function `perspective_transform` it's also possible to transform back from bird's eye view to world frame by switching self.word and self.perspective when determining the transformation matrix.

Here's an example of the warped test image with which I verified the transformation:

![Bird's view perspective transform example][warpBird]

For the future, I plan to adapt the ROI based on the image shape, but for this project setting them hardcoded in the class was sufficient for the project video.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The step for detecting lane lines in each frame consists of two parts. First the lane relevant pixels need to be determined in the binary threshold image, and second a second order polynomial has to be fitted according to the found pixels.

The lane detection is done with the call of the function `detect_lanes` in the LaneImageProcessor class line 125 in laneimageprocessor.py. The `detect_lanes` function (laneimageprocessor.py lines 326 to 382) contains the two above mentioned steps.

Finding the lane relevant pixels is a task of the LaneImageProcessor class and is done with the functions `find_lane_pixels` and - for optimization and eliminating outliners - `search_around_poly`.

For now, let's stick with the `find_lane_pixels` implementation in laneimageprocessor.py lines 238 to 324. The function takes the binary warped bird's eye view image, and has a look at the lower half. It calculates the sum of the activated pixels in each coloumn and creates a histogram for the left half and the right half of our image. The peaks in each quarter of the image then represent our starting point for a sliding window search.

Each sliding window is represented by three parameters. They have been determined through try and error:

```python
  # sliding window parameters
  number_of_windows = 8 # total number of windows vertically
  margin = 100 # pixel margin for each window left and right of center
  minpix = 50 # minimun number of pixels to detect for center replacing
```

The 8 windows with a with of 200px in total are equally distributed vertically and are shifted horizontally based on the mean of found pixels in the previous window. This is only done if at least 50 relevant pixels have been found.

If there are already detected lines from the previous cycles, the sliding window search is skipped and a search around the existing polynomial is done. This approach is implemented in the function `search_around_poly`, lines 196 to 235 in laneimageprocessor.py. Here, a margin of 100px in total is applied around the polynomial.

You can find visualization of both methods in the resulting video mentiond later in this readme. In the top left corner of the video, you can see either sliding windows (if previously not valid lanes have been detected) or a green searchpath around a polynomial (if previous lane information is present).

Here is an example image, have a look at the subimage in the top right corner, showing a sliding window search. All colored pixels are lane relevant, white pixels are discarded:

![Sliding window example image][slidingWindow]

To fit a polynomial, the class `Line`, implemented in the file line.py, is used. Each task according to line handling is implemented there, so the `detect_lanes` function does an update call to both lines left and right:

```python
  # fit left and right
  self.lines['left'].update(leftx, lefty)
  self.lines['right'].update(rightx, righty)
```

The line class then does the fit of the second order polynomial (line.py, line 77). Additionally, it does some handling besides this for the purpose of later processing (line.py, lines 83 to 105):

* keep history of the last n fits (determined by self.max_n in Line constructor)
* calculcate a simple weighted average over the last fits to get a best_fit (helps smoothen the image)
* calculate all x coordinates of the line points (y is calculated once in the constructor)
* calculate metrix for real world units (see later)

If the line has been detected successfully, it's signaled through the `self.detected` variable set to True, else False.

Here's an example for a fitted polynomial (subimage top right corner):

![Polynomial fit example][polynomialExample]

While this is working quite good with static images, adapting it to a video as a sequence of multiple frames will sooner or later lead to bad detection and implausible lines running off the road.

To address this issue, I implemented a simple sanity check for the lines, implemented in the function `sanity_check`, file lines.py, lines 220 to 260. The function simply checks:

* the absolute difference of the base position for left and right line - they should be around 3.7m according to the U.S. specification for the road in the project video, so with some margin I checked the following:

```python
  # check horizontal separation distance
  if abs(right_line.line_base_pos - left_line.line_base_pos) > 4.0:
      #print("Line base positions too far from each other")
      return False
```

* the lines should be roughly parallel - with a correct curvature and appropriate base points this should be the case when comparing different x-distances over selected y-distances:

```python
  # check lines are roughly parallel
  # if base pos and raduius of both lines are ok, it should be enough
  # to check the X distances of a few points with respect to their y positions
  # so slice the Y points into chunks and check
  chunksize = 200
  length = min(len(left_line.ally), len(right_line.ally))

  bias = None
  for i in range(0, length, chunksize):

      # take x at car as bias
      if bias is None:
          bias = abs(right_line.allx[i] - left_line.allx[i]) * left_line.xm_per_pix
      else:
          if abs(bias - abs(right_line.allx[i] - left_line.allx[i])*left_line.xm_per_pix) > 1.0:
              #print("Lines are not parallel")
              return False
```

* the radius of curvature of the lines should be roughly the same, so I checked this too:

```python
  # check curvatures -- the curvatures for left and right should be roughly
  # in the same magitude -- check for error
  if abs(left_line.radius_of_curvature - right_line.radius_of_curvature) > 200:
      #print("Line radius of curvature too different")
      return False
```

If all checks passed, the function returns True, else False. The LaneImageProcessor class then takes this information to check the handling (laneimageprocessor.py, lines 354 to 376):

```python
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
```

The class keeps track of the consecutive unplausible lines. Until a defined threshold (`self.max_unplausible_lines = 10` in the LaneImageProcessor constructor) it just asks the corresponding line objects to restore the last fit (`restore_last`, line.py, lines 131 to 187) and therefor keep the lanes in the previous position. If the threshold is exceeded, we start a completely search with calling `restore_last` with found pixels of a sliding window search.

All this is done to ignore sporadic bad frames and to recover from situations where the lanes completely run off the road. You will see this behavior in the resulting project video later.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The calculation of the radius of curvature and position is done in the Lane class' operation `calculate_metrics`, line.py, lines 190 to 216.

For the vehicle position, I first calculated the base positions for each line. This is simply calculating the x position at the lower border of the bird's view image by using the current best fit of a line:

```python
  x_center = (self.roi_warped_points[0][0] + \
      (self.roi_warped_points[1][0] - self.roi_warped_points[0][0]) / 2) * self.xm_per_pix

  # calculate the base point (near the car) with respect to the ROI
  base_fitx = self.best_fit[0]*self.roi_warped_points[2][1]**2 + \
    self.best_fit[1]*self.roi_warped_points[2][1] + \
    self.best_fit[2]
```

The `x_center` is the middle between the two bottom ROI points and describes the camera position mounted on the car. The difference between the base pos and the center, multiplied by the meters per x-pixel gives the line's base position:

```python
  # calculate the base point for the line in m according to the camera
  # as origin point
  self.line_base_pos = base_fitx * self.xm_per_pix - x_center
```

xm_per_pix is determined manually by dividing the U.S. specified lane width through the ROI positions in x direction. The same is done with the height - from the image a length of 30m has been determined with the information that dashed line segments are 3m long each:

```python
  # line base pos is calculated through the roi information
  # the used four point ROI has two points at the bottom that are straight
  # with respect to the bottom - as this points are right next to the lines,
  # they can be translated from pixels into meters with the knowledge of
  # a U.S. highway standard lane - this is an apprximation, but should be
  # good enough for this project
  # U.S. regulations minimum lane width: 3.7m
  self.xm_per_pix = 3.7 / (self.roi_warped_points[1][0] - self.roi_warped_points[0][0])

  # each dashed line is 3m long --> about 30m for warped image
  self.ym_per_pix = 35 / (self.roi_warped_points[2][1] - self.roi_warped_points[0][1])
```
The calculation is done in the Line class' constructor, line.py, lines 44 to 54.

The same is done for the radius of curvature: Through all of the x and y points of the polynomial, a new polynomial is fit - but this time with the coordinates multiplied to get real world units. After that, the radius is determined according to the formula to calculate the radius from a polynomial:


```python
  # calculate the radius of the curvature by fitting a polynom through
  # the current X and Y points in real world units
  fit_cr = np.polyfit(self.ally * self.ym_per_pix, self.allx * self.xm_per_pix, 2)

  self.radius_of_curvature = ((1 + (2*fit_cr[0]*719*self.ym_per_pix + \
      fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
```

So after calling `calculate_metrics` the real word data for each line is available. This data is then used by the LaneImageProcessor class, in the `visualize_lanes` function, file laneimageprocessor.py, lines 161 to 175:

```python
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
```

The function takes the values for each line and calculates the car position and resulting curvature radius. The values are put to the resultimage as white text:

![Polynomial fit example][polynomialExample]

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

You've already seen some of the result images above, so please have a look there.

The image itself is created through the `visualize_lanes` function, file laneimageprocessor.py, lines 131 to 193. The function takes the current frame, the resulting image from the analysis of the lane detection and the fitted polynomials to draw them into one image.

The polynomials therefore are used to fil a polygone:

```python
  # Recast the x and y points into usable format for cv2.fillPoly()
  pts_left = np.array([np.transpose(np.vstack([self.lines['left'].allx, self.lines['left'].ally]))])
  pts_right = np.array([np.flipud(np.transpose(np.vstack([self.lines['right'].allx, self.lines['right'].ally])))])
  pts = np.hstack((pts_left, pts_right))

  # Draw the lane onto the warped blank image
  cv2.fillPoly(overlay, np.int_([pts]), (0,255, 0))

  # re-warp lane_detection and overlap with current frame
  warp = self.perspective_transform('r', overlay)
  frame = cv2.addWeighted(frame, 1, warp, 0.3, 0.)
```

Again, the call of perspective_transform, but this time with the target 'r' is done, so a warping from bird's eye view to perspective view is done.

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

I did run the lane detection on all provided videos. You can find the results in the `output_videos` folder of this repository.

Here's the [link to the resulting project relevant video](./output_videos/project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The major problems I faced when implementing the project have been the following:

* It wasn't easy for me to find good thresholds to address all possible lighting and road surface conditions of the given project videos. While the project_video.mp4 itself is quite ok due to the fact that most of the time there are good road conditions, the challenge video totally fails with the same parameter set. I will improve this later on after the course by trying to use an adaptive lightness threshold in the first place and by analyzing the total lightness of the image to switch between different parameter sets for the threshold.

* Shadows on the roads and horizontal road surface lines had a huge effect on the binary threshold image. Getting the correct parameters for the thresholds wasn't easy with this. Still some parts are not good, but they are handled with the sanity check in the project video.

* When determining the ROI, I first wanted too much and did position the upper points too far. This lead to poor pixel values in the top parts most of the time, while for the lane detection there wasn't any further benefit. So I moved them back a little to the camera position, so the detection was more robust.

* Another problem were curvatures that are running out of the binary image, as the resulting polygone filled laned had a bad shape in the far distance from the car. This was also adressed by reducing the ROI distance.

The pipeline fails badly in the [challenge_video.mp4](./output_videos/challenge_video_output.mp4) because of the bad road conditions and in the [harder_challenge_video.mp4](./output_videos/harder_challenge_video_output.mp4) because of the steep curvatures and lighting conditions due to the sun, windshield reflections and shadows.

After the course I will address the issues coming up with this:
* For the challenge video, I will improve the pixel detection operation, e.g. by spliting the window vertically in more slices and do a weighted detection of the startpoint - one of the major issues is not to detect the repaired road line in the middle of the lane an the lane boarder in the middle between the roads - and to adapt a color threshold for better detecting yellow lines on light road surfaces
* For the harder challenge video, I will use the same techniques like described above for the challenge video, but additionally will widen up the ROI mask from quadliteral polygone with four points to a polygone with at least six points, so I can widen it up to the left and right. The camera has a good field of view, so detecting neighbour lines on highways and detecting sharp curvatures should be possible. Also the filter values should be tweaked.

One last point I will improve is speed. While the Python implementation, especially the numpy extension, does a fairly good job, it's much too slow. Processing the 50s project video takes about 6 to 7 minutes on my target hardware. I will address this by transforming the implementation from Python to C++. When I started the rework, I'll provide a link to the Github repository in this writeup.

So as you see, I plan to keep this repository up to date until all the videos are processed correctly. While this isn't a task according to the [Rubric](https://review.udacity.com/#!/rubrics/571/view) points and I concentrated to do the project_video.mp4 processing correctly, I want to encourage you to revisit this page soon if you are interested :-)
