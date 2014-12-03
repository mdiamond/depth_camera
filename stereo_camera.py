"""Simple demo of the stereo camera code. Creates disparity map in test_data, 
point cloud in current directory."""

import stereo_rig_model as stereo
import cv2

if __name__ == '__main__':
    lvideo = cv2.VideoCapture('test_data/videos/calibrator/left/%03d.jpeg')
    rvideo = cv2.VideoCapture('test_data/videos/calibrator/right/%03d.jpeg')
    stereo.get_stereo_depth(lvideo, rvideo,
                            {'frame': -1, 'filename': 'point_cloud.ply'})
    print 'Wrote point cloud to point_cloud.ply'
    print 'Wrote disparity map to test_data/disparity.jpg'
