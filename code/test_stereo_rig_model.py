"""Unit tests for the stereo_rig_model module."""

import os
import cv2
import stereo_rig_model as stereo
import numpy as np
import unittest


class TestStereoRigModel(unittest.TestCase):
    """Tests the functionality of the stereo_rig_model module."""

    def setUp(self):
        """Initializes shared state for unit tests."""
        self.left_video = \
            cv2.VideoCapture("test_data/videos/calibrator/test_left/012.jpeg")
        self.right_video = \
            cv2.VideoCapture("test_data/videos/calibrator/test_right/012.jpeg")

        self.expected_disparity = cv2.imread("test_data/disparity_truth.jpg")
        self.expected_pc = cv2.imread("test_data/pc_truth.ply")
        self.max_disparity_diff = 5

    def test_disparity(self):
        stereo.get_stereo_depth(self.left_video, self.right_video, {"frame": -1, "filename": "test_data/pc.ply"})
        actual_disparity = cv2.imread("test_data/disparity.jpg")
        disparity_diff = cv2.absdiff(actual_disparity, self.expected_disparity)

        # The median difference between the expected and actual disparity
        # should be less than the specified threshold.
        differences = disparity_diff.flatten().tolist()
        median_diff = sorted(differences)[len(differences) / 2]
        self.assertLessEqual(median_diff, self.max_disparity_diff)

    def test_point_cloud(self):
        stereo.get_stereo_depth(self.left_video, self.right_video, {"frame": -1, "filename": "test_data/pc.ply"})
        self.assertTrue(os.path.exists("test_data/pc.ply"))
        self.assertGreater(os.stat("test_data/pc.ply").st_size, 0)


if __name__ == "__main__":
    unittest.main()
