import cv2
import numpy as np


# FLANN matching variables
FLANN_INDEX_KDTREE = 0
TREES = 5
CHECKS = 100
KNN_ITERS = 2
LOWE_RATIO = 0.8


def _rectify_pair(sift, image_left, image_right):
    """
    Rectifies a single set of stereo images
    """
    # Get key points for both images, find matches
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

    tx1 = h1[0][2]
    ty1 = h1[1][2]
    tx2 = h2[0][2]
    ty2 = h2[1][2]

    t1 = np.array([[1, 0, -tx1], [0, 1, -ty1], [0, 0, 1]])
    t2 = np.array([[1, 0, -tx2], [0, 1, -ty2], [0, 0, 1]])

    ht1 = t1.dot(h1)
    ht2 = t2.dot(h2)

    image_left = cv2.warpPerspective(image_left,
                                     ht1
                                     (width, height))
    image_right = cv2.warpPerspective(image_right,
                                      ht2
                                      (width, height))

    return image_left, image_right


def main():
    # Open the left and right streams
    left_video = cv2.VideoCapture("test_data/videos/HNI_0054_left/%03d.png")
    right_video = cv2.VideoCapture("test_data/videos/HNI_0054_right/%03d.png")

    # Set up a SIFT feature matcher
    sift = cv2.SIFT()

    # Set up the disparity calculator
    stereo = cv2.StereoSGBM(minDisparity=16,
    numDisparities=96,
    SADWindowSize=3,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    disp12MaxDiff=1,
    P1=216,
    P2=864,
    fullDP=False)

    # Play the left video
    #ret, frame = left_video.read()
    #while ret is True:
    #    cv2.imshow("frame", frame)
    #    if cv2.waitKey(1) & 0xFF == ord('q'):
    #        break
    #    ret, frame = left_video.read()

    # Play the right video
    #ret, frame = right_video.read()
    #while ret is True:
    #    cv2.imshow("frame", frame)
    #    if cv2.waitKey(1) & 0xFF == ord('q'):
    #        break
    #    ret, frame = right_video.read()

    ret, frame_left = left_video.read()
    ret, frame_right = right_video.read()
    while ret is True:
    #frame_left, frame_right = _rectify_pair(sift,
    #                                        frame_left,
    #                                        frame_right)
        disparity = stereo.compute(frame_left,
                                    frame_right).astype(np.float32) / 16.0
        disparity = np.uint8(disparity)

        cv2.imshow('disparity', disparity)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        ret, frame_left = left_video.read()
        ret, frame_right = right_video.read()


    # Destroy all windows
    cv2.destroyAllWindows()

    # Convert the disparity image into a single channel image
    #disparity = np.uint8(disparity)

if __name__ == '__main__':
    main()
