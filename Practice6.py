#!/usr/bin/env python3
"""
sift_rotated_match.py

Detects SIFT keypoints in an image and in a rotated version of the same image,
matches them using a ratio test, and displays the best matches side by side.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate an image around its center without cropping.

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR).
    angle : float
        Rotation angle in degrees (positive = counter-clockwise).

    Returns
    -------
    np.ndarray
        Rotated image, same size as input.
    """
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)

def match_sift(img1, img2, max_matches=50):
    """
    Detect SIFT keypoints and descriptors, match between two images using
    BFMatcher + ratio test, return matched image and match count.

    Parameters
    ----------
    img1, img2 : np.ndarray
        Input images (BGR).
    max_matches : int
        Maximum number of good matches to draw.

    Returns
    -------
    matched : np.ndarray
        BGR image with matches drawn.
    count : int
        Number of matches drawn.
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # SIFT detector
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        raise RuntimeError("No descriptors found in one of the images.")

    # BFMatcher with L2 norm
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for m, n in knn:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    good = sorted(good, key=lambda x: x.distance)[:max_matches]

    matched = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        good, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return matched, len(good)

def main():
    parser = argparse.ArgumentParser(
        description="Detect and match SIFT features between an image and its rotated version."
    )
    parser.add_argument(
        "image_path",
        help="Path to the input image (e.g. test.jpg)"
    )
    parser.add_argument(
        "--angle", "-a",
        type=float,
        default=45.0,
        help="Rotation angle in degrees (default: 45)"
    )
    parser.add_argument(
        "--max", "-m",
        type=int,
        default=50,
        help="Maximum number of matches to draw (default: 50)"
    )
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    if img is None:
        print(f"Error: could not load image '{args.image_path}'", file=sys.stderr)
        sys.exit(1)

    # Create rotated version
    rotated = rotate_image(img, args.angle)

    # Match SIFT between original and rotated
    matched_img, count = match_sift(img, rotated, args.max)

    # Convert BGR to RGB for display
    matched_rgb = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB)

    # Show with Matplotlib
    plt.figure(figsize=(12, 8))
    plt.imshow(matched_rgb)
    plt.title(f"SIFT Matches (original vs. rotated {args.angle}°): {count} matches")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
