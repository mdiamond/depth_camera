import calibrator
import cv2
import numpy as np


###########
# HELPERS #
###########

def _displayDepth(name, mat):
    s = v = (np.ones(mat.shape) * 255).astype(np.uint8)
    h = ((mat - np.nanmin(mat)) / (np.nanmax(mat) - np.nanmin(mat)) * 255).astype(np.uint8)
    cv2.imshow(name, cv2.cvtColor(cv2.merge([h, s, v]), cv2.cv.CV_HSV2BGR))


def _nothing(_):
    pass


########
# MAIN #
########

def main():
    # Open the left and right streams
    left_video = cv2.VideoCapture(1)
    right_video = cv2.VideoCapture(2)

    map_1_left = np.load("test_data/map_1_left.npy")
    map_2_left = np.load("test_data/map_2_left.npy")
    map_1_right = np.load("test_data/map_1_right.npy")
    map_2_right = np.load("test_data/map_2_right.npy")

    # StereoSGBM values
    tuner_minDisparity = 10
    tuner_numDisparities = 128
    tuner_SADWindowSize = 9
    tuner_P1 = 8 * 3 * 9 * 9
    tuner_P2 = 32 * 3 * 9 * 9
    tuner_disp12MaxDiff = -1

    # Block matcher
    stereo = cv2.StereoSGBM(tuner_minDisparity, tuner_numDisparities, tuner_SADWindowSize,
                            tuner_P1, tuner_P2, tuner_disp12MaxDiff)

    cv2.destroyAllWindows()

    ret_left, frame_left = left_video.read()
    ret_right, frame_right = right_video.read()

    # Tuner GUI
    cv2.namedWindow('tuner')
    cv2.createTrackbar('minDisparity', 'tuner', tuner_minDisparity, 100, _nothing)
    cv2.createTrackbar('numDisparities', 'tuner', tuner_numDisparities, 2048, _nothing)
    cv2.createTrackbar('SADWindowSize', 'tuner', tuner_SADWindowSize, 19, _nothing)
    cv2.createTrackbar('P1', 'tuner', tuner_P1, 5000, _nothing)
    cv2.createTrackbar('P2', 'tuner', tuner_P2, 100000, _nothing)

    while ret_left is True and ret_right is True:
        #frame_left, frame_right = _rectify_stereo_pair(frame_left, frame_right)
        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        frame_left = cv2.remap(frame_left, map_1_left, map_2_left, cv2.INTER_LINEAR)
        frame_right = cv2.remap(frame_right, map_1_right, map_2_right, cv2.INTER_LINEAR)
        #cv2.imshow('left', frame_left)
        #cv2.imshow('right', frame_right)
        disparity = stereo.compute(frame_left,
                                   frame_right).astype(np.float32) / 16
        #disparity = np.uint8(disparity)
        disparity = np.float32(disparity)
        _displayDepth('tuner', disparity)
        #cv2.imshow('tuner', disparity)
        cv2.imshow('left', frame_left)
        cv2.imshow('right', frame_right)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # Update based on GUI values
        tuner_minDisparity = cv2.getTrackbarPos('minDisparity', 'tuner')
        tuner_numDisparities = max((cv2.getTrackbarPos('numDisparities', 'tuner') / 16) * 16, 16)
        tuner_SADWindowSize = cv2.getTrackbarPos('SADWindowSize', 'tuner')
        tuner_P1 = cv2.getTrackbarPos('P1', 'tuner')
        tuner_P2 = cv2.getTrackbarPos('P2', 'tuner')

        stereo = cv2.StereoSGBM(tuner_minDisparity, tuner_numDisparities, tuner_SADWindowSize,
                                tuner_P1, tuner_P2, tuner_disp12MaxDiff)

        # Get the next frame before attempting to run this loop again
        ret_left, frame_left = left_video.read()
        ret_right, frame_right = right_video.read()

    # Destroy all windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
