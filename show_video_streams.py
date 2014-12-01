import cv2
import numpy as np


###########
# HELPERS #
###########

def show(left_video, right_video):
    ret_left, frame_left = left_video.read()
    ret_right, frame_right = right_video.read()

    while ret_left is True and ret_right is True:
        cv2.imshow('frame_left', frame_left)
        cv2.imshow('frame_right', frame_right)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        ret_left, frame_left = left_video.read()
        ret_right, frame_right = right_video.read()

    cv2.destroyAllWindows()

    pass


########
# MAIN #
########

def main():
    # Open video streams for Recording
    left_video = cv2.VideoCapture(1)
    right_video = cv2.VideoCapture(2)

    # Show the streams
    show(left_video, right_video)

    # Destroy all windows
    cv2.destroyAllWindows()

    print "TERMINATING"


if __name__ == '__main__':
    main()
