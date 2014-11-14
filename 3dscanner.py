import cv2
import numpy as np


# StereoSGBM values
minDisparity = 0
numDisparities = 64
SADWindowSize = 5
P1 = 8 * 3 * SADWindowSize ** 2
P2 = 32 * 3 * SADWindowSize ** 2
disp12MaxDiff = -1
# Block matcher
stereo = cv2.StereoSGBM(minDisparity, numDisparities, SADWindowSize,
                        P1, P2, disp12MaxDiff)


def displayDepth(name, mat):
    s = v = (np.ones(mat.shape) * 255).astype(np.uint8)
    h = ((mat -np.nanmin(mat))/ (np.nanmax(mat)- np.nanmin(mat)) * 255).astype(np.uint8)
    cv2.imshow(name, cv2.cvtColor(cv2.merge([h,s,v]), cv2.cv.CV_HSV2BGR ))


def nothing(_):
    pass


def main():
    # Open the left and right streams
    left_video = cv2.VideoCapture("test_data/videos/HNI_0054_left/%03d.png")
    right_video = cv2.VideoCapture("test_data/videos/HNI_0054_right/%03d.png")

    # StereoSGBM values
    minDisparity = 0
    numDisparities = 64
    SADWindowSize = 5
    P1 = 8 * 3 * SADWindowSize ** 2
    P2 = 32 * 3 * SADWindowSize ** 2
    disp12MaxDiff = -1

    # Tuner GUI
    cv2.namedWindow('tuner')
    cv2.createTrackbar('minDisparity', 'tuner', minDisparity, 100, nothing)
    cv2.createTrackbar('numDisparities', 'tuner', numDisparities, 2048, nothing)
    cv2.createTrackbar('SADWindowSize', 'tuner', SADWindowSize, 19, nothing)
    cv2.createTrackbar('P1', 'tuner', P1, 1000, nothing)
    cv2.createTrackbar('P2', 'tuner', P2, 10000, nothing)

    # Block matcher
    stereo = cv2.StereoSGBM(minDisparity, numDisparities, SADWindowSize,
                            P1, P2, disp12MaxDiff)

    # Play the left video
    #ret, frame = left_video.read()
    #while ret is True:
        #cv2.imshow("frame", frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break
        #ret, frame = left_video.read()

    # Play the right video
    #ret, frame = right_video.read()
    #while ret is True:
        #cv2.imshow("frame", frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break
        #ret, frame = right_video.read()

    ret, frame_left = left_video.read()
    ret, frame_right = right_video.read()
    while True:
        while ret is True:
            disparity = stereo.compute(frame_left,
                                        frame_right).astype(np.float32) / 16
            disparity = np.uint8(disparity)
            cv2.imshow('tuner', disparity)
            #disparity = np.float32(disparity)
            #displayDepth('tuner', disparity)

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

            # Update based on GUI values
            minDisparity = cv2.getTrackbarPos('minDisparity', 'tuner')
            numDisparities = (cv2.getTrackbarPos('numDisparities', 'tuner') / 16) * 16
            SADWindowSize = cv2.getTrackbarPos('SADWindowSize', 'tuner')
            P1 = cv2.getTrackbarPos('P1', 'tuner')
            P2 = cv2.getTrackbarPos('P2', 'tuner')

            stereo = cv2.StereoSGBM(minDisparity, numDisparities, SADWindowSize,
                                    P1, P2, disp12MaxDiff)
           
            print minDisparity, numDisparities, SADWindowSize, P1, P2

            ret, frame_left = left_video.read()
            ret, frame_right = right_video.read()
        left_video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)
        right_video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)
        ret, frame_left = left_video.read()
        ret, frame_right = right_video.read()

    # Destroy all windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
