#!/usr/bin/env python3
"""
sift_match.py

Detects SIFT keypoints in two images of the same object under different lighting,
matches them using a ratio test, and displays the top N matches side by side.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def main(img1_path: str, img2_path: str, max_matches: int = 50):
    # 1. Load the two images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        print(f"Error: could not load images '{img1_path}' or '{img2_path}'", file=sys.stderr)
        sys.exit(1)

    # 2. Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 3. Create SIFT detector
    sift = cv2.SIFT_create()

    # 4. Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # 5. Match descriptors using BFMatcher and apply ratio test
    bf = cv2.BFMatcher()
    knn_matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in knn_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    good_matches = sorted(good_matches, key=lambda x: x.distance)[:max_matches]

    # 6. Draw the top matches
    matched_img = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # 7. Convert BGR to RGB for Matplotlib
    matched_img = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB)

    # 8. Display the result
    plt.figure(figsize=(15, 10))
    plt.imshow(matched_img)
    plt.title(f"SIFT Matches: {len(good_matches)} (max {max_matches})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Match SIFT keypoints between two images under different lighting."
    )
    parser.add_argument("img1", help="Path to the first image")
    parser.add_argument("img2", help="Path to the second image")
    parser.add_argument(
        "--max", "-m",
        type=int,
        default=50,
        help="Maximum number of matches to display (default: 50)"
    )
    args = parser.parse_args()
    main(args.img1, args.img2, args.max)
