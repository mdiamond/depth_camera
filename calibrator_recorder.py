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
    objp_right *= 0.024;
    objp_left *= 0.024;

    # Arrays to store object points and image points from all the images.
    objpoints_left = []  # 3d point in real world space
    imgpoints_left = []  # 2d points in image plane.
    objpoints_right = []  # 3d point in real world space
    imgpoints_right = []  # 2d points in image plane.

    ret_left, frame_left = left_video.read()

    shape = frame_left.shape

    i = 0
    while ret_left is True and len(imgpoints_left) < 75:
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, (6, 7), None)
        if ret_left is True:
            cv2.imwrite("test_data/videos/calibrator/left_k/%s.jpeg" % str(i).zfill(3), frame_left)
            objpoints_left.append(objp_left)
            cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            imgpoints_left.append(corners_left)
            cv2.drawChessboardCorners(frame_left, (6, 7), corners_left, ret_left)
            i = i + 1
        cv2.imshow('frame_left', frame_left)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        ret_left, frame_left = left_video.read()

    retval_left, cameraMatrix_left, distCoeffs_left, rvecs_left, tvecs_left = cv2.calibrateCamera(objpoints_left, imgpoints_left, frame_left.shape[:2])
    objpoints_left = []  # 3d point in real world space
    imgpoints_left = []  # 2d points in image plane.

    np.savetxt("test_data/k_left.txt", cameraMatrix_left)
    np.save("test_data/k_left.npy", cameraMatrix_left)

    cv2.destroyAllWindows()

    ret_right, frame_right = right_video.read()

    i = 0
    while ret_right is True and len(imgpoints_right) < 75:
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, (6, 7), None)
        if ret_right is True:
            cv2.imwrite("test_data/videos/calibrator/right_k/%s.jpeg" % str(i).zfill(3), frame_right)
            objpoints_right.append(objp_right)
            cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
            imgpoints_right.append(corners_right)
            cv2.drawChessboardCorners(frame_right, (6, 7), corners_right, ret_right)
            i = i + 1
        cv2.imshow('frame_right', frame_right)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        ret_right, frame_right = right_video.read()

    retval_right, cameraMatrix_right, distCoeffs_right, rvecs_right, tvecs_right = cv2.calibrateCamera(objpoints_right, imgpoints_right, frame_right.shape[:2])
    objpoints_right = []  # 3d point in real world space
    imgpoints_right = []  # 2d points in image plane.

    np.savetxt("test_data/k_right.txt", cameraMatrix_right)
    np.save("test_data/k_right.npy", cameraMatrix_right)

    cv2.destroyAllWindows()

    ret_left, frame_left = left_video.read()
    ret_right, frame_right = right_video.read()

    i = 0
    while ret_left is True and ret_right is True and len(imgpoints_left) < 35:
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, (6, 7), None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, (6, 7), None)

        # If found, add object points, image points (after refining them)
        if ret_left is True and ret_right is True:
            # Save the images
            cv2.imwrite("test_data/videos/calibrator/left/%s.jpeg" % str(i).zfill(3), frame_left)
            cv2.imwrite("test_data/videos/calibrator/right/%s.jpeg" % str(i).zfill(3), frame_right)

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

        ret_left, frame_left = left_video.read()
        ret_right, frame_right = right_video.read()

    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objpoints_left, imgpoints_left, imgpoints_right, shape[:2], criteria=criteria, flags=cv2.cv.CV_CALIB_USE_INTRINSIC_GUESS, cameraMatrix1=cameraMatrix_left, distCoeffs1=distCoeffs_left, cameraMatrix2=cameraMatrix_right, distCoeffs2=distCoeffs_right)
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, shape[:2], R, T)
    map_1_left, map_2_left = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, shape[:2], cv2.CV_32FC1)
    map_1_right, map_2_right = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, shape[:2], cv2.CV_32FC1)

    np.save("test_data/map_1_left.npy", map_1_left)
    np.save("test_data/map_2_left.npy", map_2_left)
    np.save("test_data/map_1_right.npy", map_1_right)
    np.save("test_data/map_2_right.npy", map_2_right)

    pass

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
