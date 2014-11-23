import cv2
import numpy as np
import StringIO


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
    cloud.write(PLY_HEADER % dict(vert_num=len(verts)))
    np.savetxt(cloud, verts, '%f %f %f %d %d %d')
    
    return cloud.getvalue()


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
    minDisparity = 8
    numDisparities = 206 / 16 * 16
    SADWindowSize = 5
    P1 = 1000
    P2 = 8200
    disp12MaxDiff = -1

    # Tuner GUI
    #cv2.namedWindow('tuner')
    #cv2.createTrackbar('minDisparity', 'tuner', minDisparity, 100, _nothing)
    #cv2.createTrackbar('numDisparities', 'tuner', numDisparities, 2048, _nothing)
    #cv2.createTrackbar('SADWindowSize', 'tuner', SADWindowSize, 19, _nothing)
    #cv2.createTrackbar('P1', 'tuner', P1, 1000, _nothing)
    #cv2.createTrackbar('P2', 'tuner', P2, 100000, _nothing)

    # Block matcher
    stereo = cv2.StereoSGBM(minDisparity, numDisparities, SADWindowSize,
                            P1, P2, disp12MaxDiff)

    ret, frame_left = left_video.read()
    ret, frame_right = right_video.read()
    #while ret is True:
    #frame_left, frame_right = _rectify_stereo_pair(frame_left, frame_right)
    #frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    #frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
    disparity = stereo.compute(frame_left,
                                frame_right).astype(np.float32) / 16
    disparity = np.uint8(disparity)
    #cv2.imshow('tuner', disparity)
    #cv2.imshow('left', frame_left)
    #cv2.imshow('right', frame_right)
    #disparity = np.float32(disparity)
    #_displayDepth('tuner', disparity)

    pc = point_cloud(disparity, frame_left, 10)

    #k = cv2.waitKey(1) & 0xFF
    #if k == 27:
        #break

    # Update based on GUI values
    #minDisparity = cv2.getTrackbarPos('minDisparity', 'tuner')
    #numDisparities = max((cv2.getTrackbarPos('numDisparities', 'tuner') / 16) * 16, 16)
    #SADWindowSize = cv2.getTrackbarPos('SADWindowSize', 'tuner')
    #P1 = cv2.getTrackbarPos('P1', 'tuner')
    #P2 = cv2.getTrackbarPos('P2', 'tuner')

    stereo = cv2.StereoSGBM(minDisparity, numDisparities, SADWindowSize,
                            P1, P2, disp12MaxDiff)
   
    print minDisparity, numDisparities, SADWindowSize, P1, P2

    # Get the next frame before attempting to run this loop again
    ret, frame_left = left_video.read()
    ret, frame_right = right_video.read()

    # Destroy all windows
    cv2.destroyAllWindows()
    with open("file.ply", 'w') as f:
        f.write(pc)
        f.close()


if __name__ == '__main__':
    main()
