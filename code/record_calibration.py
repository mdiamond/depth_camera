import cv2
import numpy as np


# Termination criteria
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


###########
# HELPERS #
###########

def record_stereo_camera(left_video, right_video):
    """
    Record stereo pairs that have known good calibration
    patterns visible into our test_data directory.
    """
    # Get the first frames
    ret_left, frame_left = left_video.read()
    ret_right, frame_right = right_video.read()

    # Get the shape
    shape = frame_left.shape

    # While there are still images to read, and we have
    # selected under a certain number images to save thus far
    i = 0
    while ret_left is True and ret_right is True and i < 15:
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, (6, 7), None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, (6, 7), None)

        # If chessboard corners are found in both images at once
        if ret_left is True and ret_right is True:
            # Save the images
            cv2.imwrite("test_data/videos/calibrator/left/%s.jpeg" % str(i).zfill(3), frame_left)
            cv2.imwrite("test_data/videos/calibrator/right/%s.jpeg" % str(i).zfill(3), frame_right)

            cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), CRITERIA)
            cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), CRITERIA)

            cv2.drawChessboardCorners(frame_left, (6, 7), corners_left, ret_left)
            cv2.drawChessboardCorners(frame_right, (6, 7), corners_right, ret_right)

        cv2.imshow('frame_left', frame_left)
        cv2.imshow('frame_right', frame_right)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

        # Decide whether or not to keep the image
        if ret_left is True and ret_right is True:
            keep = raw_input("Keep this frame? (y/n) ") == "y"
            if keep is True:
                print i
                i = i + 1

        # Advance to the next pair of frames
        ret_left, frame_left = left_video.read()
        ret_right, frame_right = right_video.read()

    cv2.destroyAllWindows()

    pass


def record_single_camera(video, name):
    """
    Record frames that have known good calibration
    patterns visible into our test_data directory.
    """
    # Get the first frame
    ret, frame = video.read()

    # Get the shape
    shape = frame.shape

    # While there are still images to read, and we have
    # selected under a certain number of images to save thus far
    i = 0
    while ret is True and i < 30:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (6, 7), None)
        # If chessboard corners are found
        if ret is True:
            # Save the image
            cv2.imwrite("test_data/videos/calibrator/%s/%s.jpeg" % (name, str(i).zfill(3)), frame)
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
            cv2.drawChessboardCorners(frame, (6, 7), corners, ret)
        cv2.imshow('frame', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        # Decide whether or not to keep the image
        if ret is True:
            keep = raw_input("Keep this frame? (y/n) ") == "y"
            if keep is True:
                print i
                i = i + 1
        # Advance to the next frame
        ret, frame = video.read()

    cv2.destroyAllWindows()

    pass


########
# MAIN #
########

def main():
    # Open video streams for Recording
    left_video = cv2.VideoCapture(1)
    right_video = cv2.VideoCapture(2)

    # Record individual cameras
    print "Recording left camera..."
    record_single_camera(left_video, "left_k")
    print "DONE"
    print "Recording right camera..."
    record_single_camera(right_video, "right_k")
    print "DONE"

    # Record stereo camera
    print "Recording stereo camera"
    record_stereo_camera(left_video, right_video)
    print "DONE"

    # Destroy all windows
    cv2.destroyAllWindows()

    print "TERMINATING"


if __name__ == '__main__':
    main()
