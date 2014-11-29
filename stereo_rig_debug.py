import calibrator
import cv2
import numpy as np
import pdb


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
minDisparity = 8
numDisparities = 206 / 16 * 16
SADWindowSize = 5
P1 = 1000
P2 = 8200
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


###########
# HELPERS #
###########

def _displayDepth(name, mat):
    s = v = (np.ones(mat.shape) * 255).astype(np.uint8)
    h = ((mat - np.nanmin(mat)) / (np.nanmax(mat) - np.nanmin(mat)) * 255).astype(np.uint8)
    cv2.imshow(name, cv2.cvtColor(cv2.merge([h, s, v]), cv2.cv.CV_HSV2BGR))


def _nothing(_):
    pass


def _rectify_stereo_pair(image_left, image_right):
    # Extract features
    sift = cv2.SIFT()
    kp_left, desc_left = sift.detectAndCompute(image_left, None)
    kp_right, desc_right = cv2.SIFT().detectAndCompute(image_right, None)
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=TREES)
    search_params = dict(checks=CHECKS)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc_left, desc_right, k=KNN_ITERS)

    # Store all the good matches as per Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < LOWE_RATIO * n.distance:
            good.append(m)

    # Pick out the left and right points from the good matches
    pts_left = np.float32(
        [kp_left[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_right = np.float32(
        [kp_right[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Compute the fundamental matrix
    F, mask = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_RANSAC)
    pts_left = pts_left[mask.ravel() == 1]
    pts_right = pts_right[mask.ravel() == 1]

    # Rectify the images
    width, height, _ = image_left.shape
    _, h1, h2 = cv2.stereoRectifyUncalibrated(
        pts_left, pts_right, F, (width, height))

    image_left = cv2.warpPerspective(image_left, h1, (height, width))
    image_right = cv2.warpPerspective(image_right, h2, (height, width))

    return image_left, image_right


########
# MAIN #
########

def main():
    # Open the left and right streams
    left_video = cv2.VideoCapture("test_data/videos/calibrator/left/%1d.jpeg")
    right_video = cv2.VideoCapture("test_data/videos/calibrator/right/%1d.jpeg")

    R1, R2, P1, P2, Q, validPixROI1, validPixROI2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = calibrator.calibrate(left_video, right_video)

    # StereoSGBM values
    tuner_minDisparity = 8
    tuner_numDisparities = 206 / 16 * 16
    tuner_SADWindowSize = 5
    tuner_P1 = 1000
    tuner_P2 = 8200
    tuner_disp12MaxDiff = -1

    # Tuner GUI
    cv2.namedWindow('tuner')
    cv2.createTrackbar('minDisparity', 'tuner', tuner_minDisparity, 100, _nothing)
    cv2.createTrackbar('numDisparities', 'tuner', tuner_numDisparities, 2048, _nothing)
    cv2.createTrackbar('SADWindowSize', 'tuner', tuner_SADWindowSize, 19, _nothing)
    cv2.createTrackbar('P1', 'tuner', tuner_P1, 1000, _nothing)
    cv2.createTrackbar('P2', 'tuner', tuner_P2, 100000, _nothing)
    cv2.createTrackbar('disp12MaxDiff', 'tuner', tuner_disp12MaxDiff, 100, _nothing)

    # Block matcher
    stereo = cv2.StereoSGBM(tuner_minDisparity, tuner_numDisparities, tuner_SADWindowSize,
                            tuner_P1, tuner_P2, tuner_disp12MaxDiff)

    left_video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)
    right_video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)

    cv2.destroyAllWindows()

    ret, frame_left = left_video.read()
    ret, frame_right = right_video.read()

    while(1):
        while ret is True:
            # frame_left, frame_right = _rectify_stereo_pair(frame_left, frame_right)
            frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
            frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
            map_1_left, map_2_left = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, frame_left.shape[:2], cv2.CV_32FC1)
            map_1_right, map_2_right = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, frame_right.shape[:2], cv2.CV_32FC1)
            frame_left = cv2.remap(frame_left, map_1_left, map_2_left, cv2.INTER_LINEAR)
            frame_right = cv2.remap(frame_right, map_1_right, map_2_right, cv2.INTER_LINEAR)
            cv2.imshow('left', frame_right)
            disparity = stereo.compute(frame_left,
                                       frame_right).astype(np.float32) / 16
            disparity = np.uint8(disparity)
            # disparity = np.float32(disparity)
            #_displayDepth('tuner', disparity)
            #cv2.imshow('tuner', disparity)
            #cv2.imshow('left', frame_left)
            #cv2.imshow('right', frame_right)

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

            # Update based on GUI values
            #tuner_minDisparity = cv2.getTrackbarPos('minDisparity', 'tuner')
            #tuner_numDisparities = max((cv2.getTrackbarPos('numDisparities', 'tuner') / 16) * 16, 16)
            #tuner_SADWindowSize = cv2.getTrackbarPos('SADWindowSize', 'tuner')
            #tuner_P1 = cv2.getTrackbarPos('P1', 'tuner')
            #tuner_P2 = cv2.getTrackbarPos('P2', 'tuner')
            #tuner_disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'tuner')

            stereo = cv2.StereoSGBM(tuner_minDisparity, tuner_numDisparities, tuner_SADWindowSize,
                                    tuner_P1, tuner_P2, tuner_disp12MaxDiff)

            print tuner_minDisparity, tuner_numDisparities, tuner_SADWindowSize, tuner_P1, tuner_P2, tuner_disp12MaxDiff

            # Get the next frame before attempting to run this loop again
            ret, frame_left = left_video.read()
            ret, frame_right = right_video.read()

        left_video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)
        right_video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)
        # Get the next frame before attempting to run this loop again
        ret, frame_left = left_video.read()
        ret, frame_right = right_video.read()

    # Destroy all windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
