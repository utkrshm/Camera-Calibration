import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def calibrate(show_pics=True):
    # Read image
    # root = os.getcwd()
    # calibration_dir = os.path.join(root, "demoImages//calibration")
    # imgs = glob.glob(os.path.join(calibrationDir, "*.jpg"))

    # Initialize
    nRows = 9
    nCols = 6
    criteria_terms = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    curr_world_pts = np.zeros((nRows * nCols, 3), np.float32)
    curr_world_pts[:, :2] = np.mgrid[0:nRows, 0:nCols].T.reshape(-1, 2)
    world_pts = []
    img_pts = []

    # find corners
    for img_path in img_pts:
        img_BGR = cv2.imread(img_path)
        img_gray = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)

        corners_found, corners_mat = cv2.findChessboardCorners(
            img_gray, (nRows, nCols), None
        )

        if corners_found:
            world_pts.append(img_path)
            refined_corners = cv2.cornerSubPix(
                img_gray, corners_mat, (11, 11), (-1, -1), criteria_terms
            )

            img_pts.append(refined_corners)

            if show_pics:
                cv2.drawChessboardCorners(
                    img_BGR, (nRows, nCols), refined_corners, corners_found
                )
                cv2.imshow("Chessboard Image (with corners)", img_BGR)
                cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Calibrate
    reprojection_error, cam_matrix, distance_coeff, rotation_vectors, translation_vectors =\
        cv2.calibrateCamera(world_pts, img_pts, img_gray.shape[::-1], None, None)
    print("Camera matrix:", cam_matrix)
    print("Reprojection error: {:.4f}".format(reprojection_error))
    
    curr_folder = os.path.dirname(os.path.abspath(__file__))    
    param_path = os.path.join(curr_folder, "calibration.npz")
    np.savez(
        param_path, 
        camMatrix = cam_matrix,
        distCoeff = distance_coeff,
        rvecs = rotation_vectors,
        tvecs = translation_vectors
    )
    
    return cam_matrix, distance_coeff


def remove_distortion(cam_matrix, distance_coeff):
    root = os.getcwd()
    img_path = os.path.join(root, "demo_images//distortion.jpeg")
    img = cv2.imread(img_path)
    
    h, w = img.shape[:2]
    
    new_cam_mat, roi = cv2.getOptimalNewCameraMatrix(cam_matrix, distance_coeff, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(img, cam_matrix, distance_coeff, None, new_cam_mat)
    
    cv2.line(img, (1769, 103), (1780, 922), (255, 255, 255), 2)
    cv2.line(undistorted_img, (1769, 103), (1780, 922), (255, 255, 255), 2)
    
    plt.figure()
    
    # Plot distorted image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    
    # Plot undistorted image
    plt.subplot(1, 2, 2)
    plt.imshow(undistorted_img)
    
    plt.show()
    
    
if __name__ == "__main__":
    cam_matrix, distance_coeffs = calibrate()
    remove_distortion(cam_matrix, distance_coeffs)
