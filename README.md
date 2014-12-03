3DSCANNER
========

Most of our code here is for testing and debugging, however, there are a few
notable modules:

* `stereo_rig_model.py` - Stereo code, generates disparity map and point cloud
* `calibrator.py` - Camera calibration for a stereo camera pair
* `stereo_camera.py` - Used to demo the stereo code

To see our stereo code in action, run:

`python stereo_camera.py`

This will run our stereo code using the saved calibrations (for our stereo camera).
This creates a point cloud (`point_cloud.ply`), and a disparity map (`test_data/disparity.jpg`).

To re-calibrate using our calibration test videos:

`python calibrator.py`

This saves the new calibration to `test_data/`, see the final report on details
on this data. Note that this only works with our stereo camera.

To use your own, see below.

# Live Stereo Disparity Map

To see a live view of our stereo code, you need to connect two cameras in a
stereo pair configuration.
We expect the two cameras to be device `1` (left) and `2` (right).

If the two cameras haven't been calibrated, things will be a bit tricky. you'll
need to print out a calibration pattern, like a chessboard pattern (located in
the [OpenCV source](https://raw.githubusercontent.com/Itseez/opencv/master/doc/pattern.png))
and record a video with the chessboard in the frame of both cameras at the same time.
Save the left and right videos somewhere in this directory, change the path to
the videos in `calibrator.py`. Then:

`python calibrator.py`

This will calibrate the cameras (provided the chessboard shows up in both frames
enough times) and save the calibration in test_data (see the final report for
details on this data).

From now on, running `stereo_camera.py` will use the saved calibration. 

`stereo_rig.py` will compute a disparity map from the input from the cameras.
You can tune the block matcher with the GUI provided.


