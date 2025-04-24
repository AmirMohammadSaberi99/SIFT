# SIFT Feature Detection and Matching Projects

A collection of **seven** Python scripts showcasing SIFT (Scale‑Invariant Feature Transform) for feature detection, description, matching, and homography estimation using OpenCV and Matplotlib.

---

## Prerequisites

- Python 3.7+
- OpenCV (with contrib modules): `pip install opencv-contrib-python`
- NumPy: `pip install numpy`
- Matplotlib: `pip install matplotlib`

---

## Projects Overview

| # | Script                              | Description                                                                                        | Usage Example                                                           |
| - | ----------------------------------- | -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| 1 | `sift_keypoints.py`                 | Detect and draw SIFT keypoints on a single image.                                                  | `python sift_keypoints.py image.jpg --max 0`                            |
| 2 | `sift_keypoint_visualization.py`    | Draw keypoints with both default and rich visualization modes.                                     | `python sift_keypoint_visualization.py img1.jpg img2.jpg`               |
| 3 | `sift_match_lighting.py`            | Match SIFT keypoints between two images under different lighting using BFMatcher + ratio test.     | `python sift_match_lighting.py img1.jpg img2.jpg --max 50`              |
| 4 | `sift_descriptors_visualization.py` | Compute and plot 128‑dim SIFT descriptors for the top keypoints alongside the image.               | `python sift_descriptors_visualization.py img.jpg`                      |
| 5 | `sift_flann_match.py`               | Match keypoints between two images using a FLANN‑based matcher and Lowe’s ratio test.              | `python sift_flann_match.py img1.jpg img2.jpg --max 50`                 |
| 6 | `sift_rotated_match.py`             | Generate a rotated version of an image and match it to the original using SIFT.                    | `python sift_rotated_match.py img.jpg --angle 60 --max 75`              |
| 7 | `sift_homography.py`                | Estimate a homography between two images using matched SIFT keypoints and warp one onto the other. | `python sift_homography.py img1.jpg img2.jpg --ratio 0.75 --reproj 5.0` |

---

## Script Details

### 1. `sift_keypoints.py`

- **Functionality:** Detects SIFT keypoints and draws them on the image (rich circles show scale & orientation).
- **CLI Flags:** `--max` to limit number of strongest keypoints.

### 2. `sift_keypoint_visualization.py`

- **Functionality:** For each input image, shows two panels: default keypoint markers and rich (oriented) markers.

### 3. `sift_match_lighting.py`

- **Functionality:** Uses BFMatcher with kNN (k=2) and Lowe’s ratio test to match between two versions of the same scene under different lighting.
- **CLI Flags:** `--max` to control how many top matches are drawn.

### 4. `sift_descriptors_visualization.py`

- **Functionality:** After detecting keypoints, sorts them by response strength and plots the 128‑bin descriptor for the top N keypoints alongside the image with keypoints.

### 5. `sift_flann_match.py`

- **Functionality:** Employs FLANN (KD‑tree) for fast nearest‑neighbor matching of SIFT descriptors, then applies a ratio test.
- **CLI Flags:** `--max` to limit number of matched lines shown.

### 6. `sift_rotated_match.py`

- **Functionality:** Rotates the original image by a user‑specified angle, then detects and matches SIFT keypoints between the original and rotated images.
- **CLI Flags:** `--angle` for rotation, `--max` for top matches.

### 7. `sift_homography.py`

- **Functionality:** Matches features between two images, uses RANSAC to estimate a 3×3 homography, draws inlier matches, and warps one image onto the other’s plane.
- **CLI Flags:** `--ratio` for Lowe’s test, `--reproj` for RANSAC reprojection threshold.

---

## How to Run

1. **Clone the repo** and install dependencies.
2. **Navigate** to the scripts folder.
3. **Run** any script, passing image paths and optional flags as shown above.

---

## License

This collection is released under the MIT License. Feel free to use and modify these scripts for research and learning purposes.

