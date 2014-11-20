import cv2
import numpy as np
import StringIO

# FLANN matching variables
FLANN_INDEX_KDTREE = 0
TREES = 5
CHECKS = 100
KNN_ITERS = 2
LOWE_RATIO = 0.8

# Set up the disparity calculator
stereo = cv2.StereoSGBM(minDisparity=22,
numDisparities=64,
SADWindowSize=10,
P1=600,
P2=2400)

# PLY file header
ply_header = '''ply
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

def get_avg_disparity(left_video, right_video):
    ret, frame_left = left_video.read()
    ret, frame_right = right_video.read()

    count = 0
    avg_disparity = np.zeros((240, 480), np.uint8)
    
    while ret is True:
        #frame_left, frame_right = _rectify_pair(sift,
        #                                        frame_left,
        #                                        frame_right)
        
        #forms average disparity
        count = count + 1
        disparity = stereo.compute(frame_left,
                                   frame_right).astype(np.float32) / 16.0
        avg_disparity = ((avg_disparity * (count-1)) + disparity)/count
        print avg_disparity

        cv2.imshow('disparity', np.uint8(avg_disparity))
        cv2.waitKey(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
           break

        ret, frame_left = left_video.read()
        ret, frame_right = right_video.read()
    
    # Destroy all windows
    cv2.destroyAllWindows()
    cv2.imshow('totaldisparity', np.uint8(avg_disparity))
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
        left_video = cv2.VideoCapture("test_data/videos/ESB/HNI_" + i + "_left/%03d.png")
        right_video = cv2.VideoCapture("test_data/videos/ESB/HNI_" + i + "_right/%03d.png")

        # Set up a SIFT feature matcher
        sift = cv2.SIFT()

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
        disparity = get_avg_disparity(left_video, right_video) #Calculates disparity average of view
        focal_length = 3
        ply_string = point_cloud(disparity, frame_left, focal_length) #forms point cloud
        with open("point_clouds/out" + i + ".ply", 'w') as f:
            f.write(ply_string)

    #JOIN POINT CLOUDS HERE:

if __name__ == '__main__':
    main()
