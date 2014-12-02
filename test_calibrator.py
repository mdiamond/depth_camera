"""Unit tests for the calibrator module."""

import cv2
import calibrator
import numpy as np
import unittest


class TestCalibrator(unittest.TestCase):
    """Tests the functionality of the calibration module."""

    def setUp(self):
        """Initializes shared state for unit tests."""
        self.left_video = \
            cv2.VideoCapture("test_data/videos/calibrator/test_left/%03d.jpeg")
        self.right_video = \
            cv2.VideoCapture("test_data/videos/calibrator/test_right/%03d.jpeg")
        
        self.left_calibration_video = \
            cv2.VideoCapture("test_data/videos/calibrator/test_left_k/%03d.jpeg")
        self.right_calibration_video = \
            cv2.VideoCapture("test_data/videos/calibrator/test_right_k/%03d.jpeg")

        self.camera_params_left = np.load("test_data/left_k_truth.npy")
        self.camera_params_right = np.load("test_data/right_k_truth.npy")

        self.map_left_1 =  np.load("test_data/map_1_left_truth.npy")
        self.map_left_2 =  np.load("test_data/map_2_left_truth.npy")
        self.map_right_1 =  np.load("test_data/map_1_right_truth.npy")
        self.map_right_2 =  np.load("test_data/map_2_right_truth.npy")

        self.max_diff = 0.1

    def _calib_diff(self, expected, actual):
        self.assertEqual(len(expected), len(actual))
        diff = 0
        for expected, actual in zip(expected, actual):
            diff += sum(np.abs(expected - actual))
        return diff

    def test_calibrate_single_camera(self):
        self.camera_matrix_left, self.distcoeffs_left = \
            calibrator.calibrate_single_camera(self.left_calibration_video,
                                               "left_k")

        self.camera_matrix_right, self.distcoeffs_right = \
            calibrator.calibrate_single_camera(self.right_calibration_video,
                                               "right_k")

        actual_camera_params_left = np.load("test_data/left_k.npy")
        actual_camera_params_right = np.load("test_data/right_k.npy")
        self.assertLessEqual(self._calib_diff(self.camera_params_left,
                             actual_camera_params_left), self.max_diff)

        self.assertLessEqual(self._calib_diff(self.camera_params_right,
                             actual_camera_params_right), self.max_diff)

    def test_calibrate_stereo_camera(self):
        camera_matrix_left, distcoeffs_left = \
            calibrator.calibrate_single_camera(self.left_calibration_video,
                                               "left_k")

        camera_matrix_right, distcoeffs_right = \
            calibrator.calibrate_single_camera(self.right_calibration_video,
                                               "right_k")

        calibrator.calibrate_stereo_camera(self.left_video, self.right_video,
                                           camera_matrix_left,
                                           distcoeffs_left,
                                           camera_matrix_right,
                                           distcoeffs_right)

        map_1_left = np.load("test_data/map_1_left.npy")
        map_2_left = np.load("test_data/map_2_left.npy")
        map_1_right = np.load("test_data/map_1_right.npy")
        map_2_right = np.load("test_data/map_2_right.npy")

        self.assertLessEqual(self._calib_diff(self.map_left_1, map_1_left),
                             self.max_diff)
        self.assertLessEqual(self._calib_diff(self.map_left_2, map_2_left),
                             self.max_diff)
        self.assertLessEqual(self._calib_diff(self.map_right_2, map_2_right),
                             self.max_diff)
        self.assertLessEqual(self._calib_diff(self.map_right_2, map_2_right),
                             self.max_diff)


if __name__ == '__main__':
    unittest.main()
