import cv2
import numpy as np


###########
# HELPERS #
###########

def calibrate(left_video, right_video):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp_left = np.zeros((6 * 7, 3), np.float32)
    objp_left[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
    objp_right = np.zeros((6 * 7, 3), np.float32)
    objp_right[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints_left = []  # 3d point in real world space
    imgpoints_left = []  # 2d points in image plane.
    objpoints_right = []  # 3d point in real world space
    imgpoints_right = []  # 2d points in image plane.

    ret_left, frame_left = left_video.read()
    ret_right, frame_right = right_video.read()

    i = 0
    while ret_left is True and ret_right is True and len(imgpoints_left) < 10:
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, (6, 7), None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, (6, 7), None)

        # If found, add object points, image points (after refining them)
        print ret_left
        print ret_right
        if ret_left is True and ret_right is True:
            # Save the images
            cv2.imwrite('test_data/videos/calibrator/left/{num}.jpeg'.format(num=i), frame_left)
            cv2.imwrite('test_data/videos/calibrator/right/{num}.jpeg'.format(num=i), frame_right)

            # Append object points
            objpoints_left.append(objp_left)
            objpoints_right.append(objp_right)

            # Refine chessboard corner locations
            cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

            # Append image points
            imgpoints_left.append(corners_left)
            imgpoints_right.append(corners_right)

            # Draw and display the corners
            cv2.drawChessboardCorners(frame_left, (6, 7), corners_left, ret_left)
            cv2.drawChessboardCorners(frame_right, (6, 7), corners_right, ret_right)

            i = i + 1

        cv2.imshow('frame_left', frame_left)
        cv2.imshow('frame_right', frame_right)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        ret_right, frame_left = left_video.read()
        ret_left, frame_right = right_video.read()

    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objpoints_left, imgpoints_left, imgpoints_right, frame_left.shape[: 2], criteria=criteria, flags=cv2.cv.CV_CALIB_SAME_FOCAL_LENGTH)
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, frame_left.shape[: 2], R, T)

    return R1, R2, P1, P2, Q, validPixROI1, validPixROI2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F


########
# MAIN #
########

def main():
    # Open the left and right streams
    left_video = cv2.VideoCapture(1)
    right_video = cv2.VideoCapture(2)

    # Calibrate
    calibrate(left_video, right_video)

    # Destroy all windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
