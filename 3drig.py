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


###########
# HELPERS #
##########

def _displayDepth(name, mat):
    s = v = (np.ones(mat.shape) * 255).astype(np.uint8)
    h = ((mat -np.nanmin(mat))/ (np.nanmax(mat)- np.nanmin(mat)) * 255).astype(np.uint8)
    cv2.imshow(name, cv2.cvtColor(cv2.merge([h,s,v]), cv2.cv.CV_HSV2BGR ))


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
    left_video = cv2.VideoCapture(1)
    right_video = cv2.VideoCapture(2)

    # StereoSGBM values
    minDisparity = 0
    numDisparities = 64
    SADWindowSize = 5
    P1 = 8 * 3 * SADWindowSize ** 2
    P2 = 32 * 3 * SADWindowSize ** 2
    disp12MaxDiff = -1

    # Tuner GUI
    cv2.namedWindow('tuner')
    cv2.createTrackbar('minDisparity', 'tuner', minDisparity, 100, _nothing)
    cv2.createTrackbar('numDisparities', 'tuner', numDisparities, 2048, _nothing)
    cv2.createTrackbar('SADWindowSize', 'tuner', SADWindowSize, 19, _nothing)
    cv2.createTrackbar('P1', 'tuner', P1, 1000, _nothing)
    cv2.createTrackbar('P2', 'tuner', P2, 100000, _nothing)

    # Block matcher
    stereo = cv2.StereoSGBM(minDisparity, numDisparities, SADWindowSize,
                            P1, P2, disp12MaxDiff)

    ret, frame_left = left_video.read()
    ret, frame_right = right_video.read()
    while True:
        while ret is True:
            #frame_left, frame_right = _rectify_stereo_pair(frame_left, frame_right)
            frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
            frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
            disparity = stereo.compute(frame_left,
                                        frame_right).astype(np.float32) / 16
            disparity = np.uint8(disparity)
            cv2.imshow('tuner', disparity)
            cv2.imshow('left', frame_left)
            cv2.imshow('right', frame_right)
            #disparity = np.float32(disparity)
            #_displayDepth('tuner', disparity)

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

            # Update based on GUI values
            minDisparity = cv2.getTrackbarPos('minDisparity', 'tuner')
            numDisparities = max((cv2.getTrackbarPos('numDisparities', 'tuner') / 16) * 16, 16)
            SADWindowSize = cv2.getTrackbarPos('SADWindowSize', 'tuner')
            P1 = cv2.getTrackbarPos('P1', 'tuner')
            P2 = cv2.getTrackbarPos('P2', 'tuner')

            stereo = cv2.StereoSGBM(minDisparity, numDisparities, SADWindowSize,
                                    P1, P2, disp12MaxDiff)
           
            print minDisparity, numDisparities, SADWindowSize, P1, P2

            # Get the next frame before attempting to run this loop again
            ret, frame_left = left_video.read()
            ret, frame_right = right_video.read()

        # Restart the video
        left_video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)
        right_video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)
        ret, frame_left = left_video.read()
        ret, frame_right = right_video.read()

    # Destroy all windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
