#!/usr/bin/env python3
"""
sift_keypoints.py

Loads a single image, detects SIFT keypoints, draws them on the image, 
and displays the result.
"""

import cv2  
import argparse
import sys
import matplotlib.pyplot as plt

def main(image_path: str, max_keypoints: int = 0):
    # 1. Load image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: could not load image '{image_path}'", file=sys.stderr)
        sys.exit(1)

    # 2. Convert to grayscale
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 3. Create SIFT detector
    sift = cv2.SIFT_create()

    # 4. Detect keypoints and compute descriptors (we only need keypoints here)
    keypoints = sift.detect(img_gray, None)

    # 5. (Optional) Limit number of keypoints
    if max_keypoints > 0 and len(keypoints) > max_keypoints:
        keypoints = sorted(keypoints, key=lambda kp: -kp.response)[:max_keypoints]

    # 6. Draw keypoints on the original image
    img_with_kp = cv2.drawKeypoints(
        img_bgr, 
        keypoints, 
        None, 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # 7. Convert BGR->RGB for Matplotlib and display
    img_rgb = cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.title(f"SIFT Keypoints (detected: {len(keypoints)})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect and draw SIFT keypoints on an image."
    )
    parser.add_argument(
        "image_path",
        help="Path to the input image (e.g. test.png)"
    )
    parser.add_argument(
        "--max", "-m",
        type=int,
        default=0,
        help="Maximum number of strongest keypoints to draw (default: all)"
    )
    args = parser.parse_args()

    main(args.image_path, args.max)
