#!/usr/bin/env python3
"""
sift_keypoint_visualization.py

Detects SIFT keypoints on one or more images and displays two visualizations side by side:
1) Default keypoint drawing (small circles)
2) Rich keypoint drawing (circles sized & oriented)
"""

import cv2
import matplotlib.pyplot as plt
import argparse
import sys

def draw_keypoints(img_bgr, keypoints, rich=False):
    """
    Draw keypoints on a BGR image.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input color image in BGR.
    keypoints : list of cv2.KeyPoint
        Detected keypoints.
    rich : bool
        If True, use DRAW_RICH_KEYPOINTS; otherwise default flags.
    """
    flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS if rich else 0
    return cv2.drawKeypoints(img_bgr, keypoints, None, flags=flags)

def process_and_plot(image_paths):
    sift = cv2.SIFT_create()
    n = len(image_paths)
    fig, axes = plt.subplots(n, 2, figsize=(10, 5 * n))
    if n == 1:
        axes = [axes]

    for row, path in enumerate(image_paths):
        # Load and check
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            print(f"Error: Could not load '{path}'", file=sys.stderr)
            continue

        # Detect keypoints
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        keypoints = sift.detect(gray, None)

        # Draw default and rich
        img_def = draw_keypoints(img_bgr, keypoints, rich=False)
        img_rich = draw_keypoints(img_bgr, keypoints, rich=True)

        # Convert to RGB for Matplotlib
        img_def_rgb = cv2.cvtColor(img_def, cv2.COLOR_BGR2RGB)
        img_rich_rgb = cv2.cvtColor(img_rich, cv2.COLOR_BGR2RGB)

        # Plot
        axes[row][0].imshow(img_def_rgb)
        axes[row][0].set_title(f"{path}\nDefault Keypoints ({len(keypoints)})")
        axes[row][0].axis('off')

        axes[row][1].imshow(img_rich_rgb)
        axes[row][1].set_title("Rich Keypoints")
        axes[row][1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Draw SIFT keypoints using default and rich visualizations."
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Paths to input images (e.g. Test.jpg Test2.jpg)"
    )
    args = parser.parse_args()
    process_and_plot(args.images)
