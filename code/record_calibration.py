import cv2
import numpy as np


# Termination criteria
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


###########
# HELPERS #
###########

def record_stereo_camera(left_video, right_video):
    ret_left, frame_left = left_video.read()
    ret_right, frame_right = right_video.read()

    shape = frame_left.shape

    i = 0
    while ret_left is True and ret_right is True and i < 15:
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, (6, 7), None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, (6, 7), None)

        if ret_left is True and ret_right is True:

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

        if ret_left is True and ret_right is True:
            keep = raw_input("Keep this frame? (y/n) ") == "y"
            if keep is True:
                print i
                i = i + 1

        ret_left, frame_left = left_video.read()
        ret_right, frame_right = right_video.read()

    cv2.destroyAllWindows()

    pass


def record_single_camera(video, name):
    ret, frame = video.read()

    shape = frame.shape

    i = 0
    while ret is True and i < 30:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (6, 7), None)
        if ret is True:
            cv2.imwrite("test_data/videos/calibrator/%s/%s.jpeg" % (name, str(i).zfill(3)), frame)
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
            cv2.drawChessboardCorners(frame, (6, 7), corners, ret)
        cv2.imshow('frame', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if ret is True:
            keep = raw_input("Keep this frame? (y/n) ") == "y"
            if keep is True:
                print i
                i = i + 1
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
