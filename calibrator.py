"""Camera calibration. Calibrates a single camera and a stereo camera.
"""
import cv2
import numpy as np


# Termination criteria
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


###########
# HELPERS #
###########

def calibrate_stereo_camera(left_video, right_video, cameraMatrix_left,
                            distCoeffs_left, cameraMatrix_right,
                            distCoeffs_right):
    """Calibrate a stereo camera pair."""
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp_left = np.zeros((6 * 7, 3), np.float32)
    objp_left[:, :2] = np.mgrid[0:6, 0:7].T.reshape(-1, 2)
    objp_right = np.zeros((6 * 7, 3), np.float32)
    objp_right[:, :2] = np.mgrid[0:6, 0:7].T.reshape(-1, 2)
    objp_left *= 0.024
    objp_right *= 0.024

    # Arrays to store object points and image points from all the images.
    objpoints_left = []  # 3d point in real world space
    imgpoints_left = []  # 2d points in image plane.
    objpoints_right = []  # 3d point in real world space
    imgpoints_right = []  # 2d points in image plane.

    ret_left, frame_left = left_video.read()
    ret_right, frame_right = right_video.read()

    shape = frame_left.shape

    i = 0
    while ret_left is True and ret_right is True and i < 35:
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, (6, 7), None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, (6, 7), None)

        # If found, add object points, image points (after refining them)
        if ret_left is True and ret_right is True:

            # Append object points
            objpoints_left.append(objp_left)
            objpoints_right.append(objp_right)

            # Refine chessboard corner locations
            cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), CRITERIA)
            cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), CRITERIA)

            # Append image points
            imgpoints_left.append(corners_left)
            imgpoints_right.append(corners_right)

            # Draw and display the corners
            cv2.drawChessboardCorners(frame_left, (6, 7), corners_left, ret_left)
            cv2.drawChessboardCorners(frame_right, (6, 7), corners_right, ret_right)

            print i
            i = i + 1

        cv2.imshow('frame_left', frame_left)
        cv2.imshow('frame_right', frame_right)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        ret_left, frame_left = left_video.read()
        ret_right, frame_right = right_video.read()

    print "Calculating calibration..."
    stereo_calibrate_args = {
        'objectPoints': objpoints_left,
        'imagePoints1': imgpoints_left,
        'imagePoints2': imgpoints_right,
        'imageSize': (shape[1], shape[0]),
        'criteria': CRITERIA,
        'flags': cv2.cv.CV_CALIB_USE_INTRINSIC_GUESS,
        'cameraMatrix1': cameraMatrix_left,
        'cameraMatrix2': cameraMatrix_right,
        'distCoeffs1': distCoeffs_left,
        'distCoeffs2': distCoeffs_right
    }
    _, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
                                    cv2.stereoCalibrate(**stereo_calibrate_args)
    print "Calculating rectification..."
    stereo_rectify_args = {
        'cameraMatrix1': cameraMatrix1,
        'cameraMatrix2': cameraMatrix2,
        'distCoeffs1': distCoeffs1,
        'distCoeffs2': distCoeffs2,
        'imageSize': (shape[1], shape[0]),
        'R': R,
        'T': T
    }
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = \
                                        cv2.stereoRectify(**stereo_rectify_args)
    print "Calculating mappings..."
    map_1_left, map_2_left = cv2.initUndistortRectifyMap(cameraMatrix1,
                                                         distCoeffs1, R1, P1,
                                                         (shape[1], shape[0]),
                                                         cv2.CV_32FC1)
    map_1_right, map_2_right = cv2.initUndistortRectifyMap(cameraMatrix2,
                                                           distCoeffs2, R2, P2,
                                                           (shape[1], shape[0]),
                                                           cv2.CV_32FC1)

    np.save("test_data/map_1_left.npy", map_1_left)
    np.save("test_data/map_2_left.npy", map_2_left)
    np.save("test_data/map_1_right.npy", map_1_right)
    np.save("test_data/map_2_right.npy", map_2_right)


def calibrate_single_camera(video, name):
    """Find intrinsic parameters for a single camera."""
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:6, 0:7].T.reshape(-1, 2)
    objp *= 0.024

    objpoints = []
    imgpoints = []

    ret, frame = video.read()

    shape = frame.shape

    i = 0
    while ret is True and i < 75:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (6, 7), None)
        if ret is True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(frame, (6, 7), corners, ret)
            print i
            i = i + 1
        cv2.imshow('frame', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        ret, frame = video.read()

    print "Calculating calibration..."
    retval, cameraMatrix, distCoeffs, rvecs, tvecs = \
                            cv2.calibrateCamera(objpoints, imgpoints, shape[:2])

    np.savetxt("test_data/%s.txt" % name, cameraMatrix)
    np.save("test_data/%s.npy" % name, cameraMatrix)

    cv2.destroyAllWindows()

    return cameraMatrix, distCoeffs


########
# MAIN #
########

def main():
    # Open video streams for individual calibration
    left_video = cv2.VideoCapture("test_data/videos/calibrator/left_k/%03d.jpeg")
    right_video = cv2.VideoCapture("test_data/videos/calibrator/right_k/%03d.jpeg")

    # Calibrate individual cameras
    print "Calibrating left camera..."
    cameraMatrix_left, distCoeffs_left = calibrate_single_camera(left_video, "left_k")
    print "DONE"
    print "Calibrating right camera..."
    cameraMatrix_right, distCoeffs_right = calibrate_single_camera(right_video, "right_k")
    print "DONE"

    # Open video streams for stereo calibration
    left_video = cv2.VideoCapture("test_data/videos/calibrator/left/%03d.jpeg")
    right_video = cv2.VideoCapture("test_data/videos/calibrator/right/%03d.jpeg")

    # Calibrate stereo camera
    print "Calibrating stereo camera..."

    calibrate_stereo_camera(left_video, right_video, cameraMatrix_left,
                            distCoeffs_left, cameraMatrix_right,
                            distCoeffs_right)
    print "DONE"

    # Destroy all windows
    cv2.destroyAllWindows()

    print "TERMINATING"


if __name__ == '__main__':
    main()
