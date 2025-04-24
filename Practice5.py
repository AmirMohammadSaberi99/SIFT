#!/usr/bin/env python3
"""
sift_flann_match.py

Detects SIFT keypoints in two images and matches them using a FLANN-based matcher.
Displays the top matches side by side.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def flann_match(img1_path: str, img2_path: str, max_matches: int = 50):
    # 1. Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        raise FileNotFoundError(f"Could not load '{img1_path}' or '{img2_path}'")

    # 2. Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 3. Create SIFT detector and detect keypoints + descriptors
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        print("No descriptors found in one of the images.", file=sys.stderr)
        return None

    # 4. Set up FLANN matcher parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 5. Perform KNN matching with k=2
    knn_matches = flann.knnMatch(des1, des2, k=2)

    # 6. Apply Lowe's ratio test
    good = []
    for m, n in knn_matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # 7. Sort by distance and keep top matches
    good = sorted(good, key=lambda x: x.distance)[:max_matches]

    # 8. Draw matches
    matched_img = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        good, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return matched_img, len(good)

def main():
    parser = argparse.ArgumentParser(
        description="Match SIFT keypoints between two images using FLANN."
    )
    parser.add_argument("img1", help="Path to first image")
    parser.add_argument("img2", help="Path to second image")
    parser.add_argument(
        "--max", "-m",
        type=int,
        default=50,
        help="Maximum number of matches to display (default=50)"
    )
    args = parser.parse_args()

    result = flann_match(args.img1, args.img2, args.max)
    if result is None:
        sys.exit(1)

    matched_img, count = result
    # Convert BGR to RGB for Matplotlib
    matched_img = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 10))
    plt.imshow(matched_img)
    plt.title(f"FLANN SIFT Matches: {count}/{args.max}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
