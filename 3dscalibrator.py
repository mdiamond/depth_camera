import cv2
import numpy as np


#############
# CONSTANTS #
#############

# FLANN matching variables
FLANN_INDEX_KDTREE = 0
TREES = 5
CHECKS = 100
KNN_ITERS = 2
LOWE_RATIO = 0.8
# StereoSGBM values
minDisparity = 0
numDisparities = 64
SADWindowSize = 5
P1 = 8 * 3 * SADWindowSize ** 2
P2 = 32 * 3 * SADWindowSize ** 2
disp12MaxDiff = -1
# The header for a PLY point cloud
PLY_HEADER = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''


########
# MAIN #
########

def main():
    # Open the left and right streams
    left_video = cv2.VideoCapture("test_data/videos/HNI_0060_left/%03d.png")
    right_video = cv2.VideoCapture("test_data/videos/HNI_0060_right/%03d.png")

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp_left = np.zeros((6*7,3), np.float32)
    objp_left[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
    objp_right = np.zeros((6*7,3), np.float32)
    objp_right[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints_left = [] # 3d point in real world space
    imgpoints_left = [] # 2d points in image plane.
    objpoints_right = [] # 3d point in real world space
    imgpoints_right = [] # 2d points in image plane.

    ret, frame_left = left_video.read()
    ret, frame_right = right_video.read()
    while ret is True:
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, (8, 8), None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, (8, 8), None)

        # If found, add object points, image points (after refining them)
        print ret_left
        if ret_left == True:
            objpoints_left.append(objp_left)

            corners_left_2 = cv2.cornerSubPix(gray_left, corners_left, (11,11),(-1,-1),criteria)
            imgpoints_left.append(corners_left_2)

            # Draw and display the corners
            frame_left = cv2.drawChessboardCorners(frame_left, (8, 8), corners_left_2, ret_left)
            cv2.imshow('frame_left', frame_left)
            cv2.waitKey()

        print ret_right
        if ret_right == True:
            objpoints_right.append(objp_right)

            corners_right_2 = cv2.cornerSubPix(gray_right, corners_right, (11,11),(-1,-1), criteria)
            imgpoints_right.append(corners_right_2)

            # Draw and display the corners
            frame_right = cv2.drawChessboardCorners(frame_right, (8, 8), corners_right_2, ret_right)
            cv2.imshow('frame_right', frame_right)
            cv2.waitKey()

        ret, frame_left = left_video.read()
        ret, frame_right = right_video.read()

    # Destroy all windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
