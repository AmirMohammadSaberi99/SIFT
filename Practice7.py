#!/usr/bin/env python3
"""
sift_homography.py

Detects SIFT keypoints in two images, matches them with Lowe’s ratio test,
estimates a homography using RANSAC, warps one image into the frame of the
other, and displays the matched keypoints and the warped result.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def detect_and_match_sift(img1_gray, img2_gray, ratio_thresh=0.75):
    """
    Detect SIFT keypoints & descriptors in two grayscale images,
    match with KNN + ratio test, and return good matches and keypoints.
    """
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)
    if des1 is None or des2 is None:
        return [], kp1, kp2

    # FLANN parameters
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn_matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good.append(m)
    return good, kp1, kp2

def estimate_homography(kp1, kp2, matches, reproj_thresh=5.0):
    """
    Estimate homography from matched keypoints using RANSAC.
    Returns the 3×3 homography matrix and mask of inliers.
    """
    if len(matches) < 4:
        return None, None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, reproj_thresh)
    return H, mask

def draw_matches(img1, img2, kp1, kp2, matches, mask):
    """
    Draw only the inlier matches (mask==1) between img1 and img2.
    """
    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=None,
        matchesMask=mask.ravel().tolist(),
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
    )
    return cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)

def warp_image(img, H, dst_shape):
    """
    Warp img by homography H into a canvas of size dst_shape (h, w).
    """
    h, w = dst_shape
    return cv2.warpPerspective(img, H, (w, h))

def main(img1_path, img2_path, ratio=0.75, reproj_thresh=5.0):
    # 1. Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        print(f"Error: cannot load '{img1_path}' or '{img2_path}'", file=sys.stderr)
        sys.exit(1)

    # 2. Grayscale conversions
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 3. Detect & match
    matches, kp1, kp2 = detect_and_match_sift(gray1, gray2, ratio)

    # 4. Estimate homography
    H, mask = estimate_homography(kp1, kp2, matches, reproj_thresh)
    if H is None:
        print("Not enough matches for homography.", file=sys.stderr)
        sys.exit(1)

    # 5. Draw inlier matches
    matched_img = draw_matches(img1, img2, kp1, kp2, matches, mask)

    # 6. Warp image1 into image2’s frame
    h2, w2 = img2.shape[:2]
    warped = warp_image(img1, H, (h2, w2))

    # 7. Display results
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
    plt.title("Inlier SIFT Matches & Homography")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    # Overlay warped img1 onto img2
    overlay = img2.copy()
    mask_warp = (warped.sum(axis=2) > 0)
    overlay[mask_warp] = warped[mask_warp]
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Warped Image1 Over Image2")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and visualize homography between two images using SIFT."
    )
    parser.add_argument("img1", help="First image path (source)")
    parser.add_argument("img2", help="Second image path (destination)")
    parser.add_argument(
        "--ratio", "-r",
        type=float,
        default=0.75,
        help="Lowe's ratio for match filtering (default=0.75)"
    )
    parser.add_argument(
        "--reproj", "-p",
        type=float,
        default=5.0,
        help="RANSAC reprojection threshold (default=5.0)"
    )
    args = parser.parse_args()

    main(args.img1, args.img2, args.ratio, args.reproj)
