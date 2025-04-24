#!/usr/bin/env python3
"""
sift_descriptors_visualization.py

Loads one or more images, detects SIFT keypoints & descriptors, draws
the keypoints on the image, and plots descriptor histograms for the
top 5 keypoints by strength.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def visualize_sift_descriptors(image_path: str, num_plot: int = 5):
    # 1. Load image and gray-scale
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load '{image_path}'")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Create SIFT, detect keypoints + descriptors
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if len(keypoints) == 0:
        print(f"No SIFT keypoints found in '{image_path}'", file=sys.stderr)
        return

    # 3. Sort by response and keep top num_plot
    sorted_idx = np.argsort([-kp.response for kp in keypoints])
    top_idx = sorted_idx[:num_plot]

    # 4. Draw all keypoints (rich)
    img_kp = cv2.drawKeypoints(
        img_bgr, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    img_kp_rgb = cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB)

    # 5. Plot
    fig = plt.figure(constrained_layout=True, figsize=(4*(num_plot), 8))
    gs = fig.add_gridspec(2, num_plot)

    # 5a. Show image with keypoints spanning all columns in row 0
    ax_img = fig.add_subplot(gs[0, :])
    ax_img.imshow(img_kp_rgb)
    ax_img.set_title(f"{image_path} — SIFT Keypoints ({len(keypoints)})")
    ax_img.axis('off')

    # 5b. For each top keypoint, plot its 128-bin descriptor
    for col, idx in enumerate(top_idx):
        ax = fig.add_subplot(gs[1, col])
        desc = descriptors[idx]
        ax.bar(np.arange(128), desc, width=1.0)
        ax.set_title(f"KP#{idx}\nSize={int(keypoints[idx].size)}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def main(paths):
    for p in paths:
        visualize_sift_descriptors(p, num_plot=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect SIFT keypoints and plot descriptor histograms."
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Paths to one or more input images"
    )
    args = parser.parse_args()
    main(args.images)
