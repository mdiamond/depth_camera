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
    left_video = cv2.VideoCapture("test_data/videos/ESB/HNI_0_left/%03d.png")
    right_video = cv2.VideoCapture("test_data/videos/ESB/HNI_0_right/%03d.png")

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
    cv2.imshow('totaldisparity', avg_disparity)
    cv2.waitKey(0)

    return avg_disparity

def point_cloud(disparity_image, image_left, focal_length):
    """Create a point cloud from a disparity image and a focal length.
        Arguments:
        disparity_image: disparities in pixels.
        image_left: BGR-format left stereo image, to color the points.
        focal_length: the focal length of the stereo camera, in pixels.
        Returns:
        A string containing a PLY point cloud of the 3D locations of the
        pixels, with colors sampled from left_image. You may filter low-
        disparity pixels or noise pixels if you choose.
        """
    h, w = image_left.shape[:2]
    Q = np.float32([[1, 0,  0, w / 2],
                    [0, -1,  0,  h / 2],  # turn points 180 deg around x-axis,
                    [0, 0, focal_length,  0],  # so that y-axis looks up
                    [0, 0,  0,  1]])
                    
    # reproject image points to 3D space, compute the colors
    points = cv2.reprojectImageTo3D(disparity_image, Q)
    colors = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)
    mask = disparity_image > disparity_image.min()
    out_points = points[mask]
    out_colors = colors[mask]
    
    # write PLY string data to StringIO object and return the contents
    cloud = StringIO.StringIO()
    verts = np.hstack([out_points, out_colors])
    cloud.write(ply_header % dict(vert_num=len(verts)))
    np.savetxt(cloud, verts, '%f %f %f %d %d %d')
    
    return cloud.getvalue()

def main():
    for i in ["0", "45", "90", "135", "180", "225", "270", "315"]:
        # Open the left and right streams
        left_video = cv2.VideoCapture("test_data/videos/HNI_" + i + "0054_left/$03d.png")
        right_video = cv2.VideoCapture("test_data/videos/HNI_" + i + "0054_right/%03d.png")

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
        disparity = get_avg_disparity(left_video, right_video) #Calculates disparity average of view
        
        #focal length must be computed and inserted here
        ply_string = point_cloud(disparity, frame_left, focal_length) #forms point cloud

if __name__ == '__main__':
    main()
