# Computer Vision Homeworks (CSc 8830)

Hosted on : https://cv-homeworkmodule-1.onrender.com/


## Structure

```
app.py                          
homeworks/
  hw2_calibration/               # Homework 2 - camera calibration & perspective 
  hw3_blurring/                  # Homework 3 - blurring: spatial filter vs. Fourier domain
    
  hw4_segmentation/               # Homework 4 - classical human boundary segmentation vs. SAM2
  hw6_motion/                     # Homework 6 - optical flow, Lucas-Kanade tracking, structure from motion
templates/
static/

```

## Homework 2 - Camera Calibration & Measurement

Calibrates a camera from chessboard photos, then uses the intrinsics plus a known camera-to-object distance to
turn a pixel measurement into a real-world length. Validated against a 20+ object ground-truth CSV.

## Homework 3 - Image Blurring: Spatial Filter vs. Fourier Domain

Blurs an uploaded image two independent ways on the same kernel:
- **Spatial**: direct 2D convolution 
- **Frequency**: zero-pad the image and kernel, multiply their FFTs, inverse-FFT, crop back (the standard
  "linear convolution via padded circular convolution" trick, needed because a bare FFT multiply gives circular,
  not linear, convolution).



## Homework 4 - Human Boundary Segmentation (Classical CV) vs. SAM2

Two classical (non-ML, non-DL) OpenCV segmentation pipelines, each with a boundary/contour output:
- **RGB**: GrabCut (iterative graph-cut / GMM energy minimization on a single image, no trained model) seeded
  with a rectangle you drag on the photo, or an automatic whole-image box.
- **Thermal**: Otsu global thresholding + morphology - a person is almost always the warmest (brightest) blob in
  a thermal frame.



<!-- ## Homework 6 - Optical Flow, Tracking and Structure from Motion -->

<!-- Three pages under `/hw6`, plus a theory page with the derivations:

- **Optical flow video**: upload a video and pick a 30 s window. Dense Farneback flow is computed between
  every pair of frames and written back out as a video (arrows on the frame | colour-coded flow). The same pass
  fits a global affine motion to each frame to separate camera pan / zoom / roll from objects moving on their
  own, and turns the numbers into observations backed by plots (motion over time, direction histogram, motion
  heat maps, busiest frame).
- **Two-frame tracking**: a pyramidal Lucas-Kanade tracker written from the equations (`lucas_kanade.py`,
  including the bilinear interpolation) tracks corners between two consecutive frames. Every prediction is
  compared with the actual location found by NCC template matching, with points you click by hand, and with
  OpenCV's `calcOpticalFlowPyrLK`. One feature is worked through iteration by iteration.
- **Structure from motion**: four photos of a flat object, with the same boundary points clicked in each.
  Normalised DLT homographies (view 1 to views 2-4) are decomposed into R, t/d, N. The right solution is picked
  by positive depth plus a shared plane normal. Then the points are triangulated from all four views, refined
  with bundle adjustment, and scaled to cm with one known length, which gives the boundary, edge lengths, area
  and camera positions. All the intermediate matrices are printed, and a JSON report (camera parameters, EXIF,
  clicks, poses) can be downloaded.

The default intrinsics for part 3 are from the HW2 calibration photos in `calibration_images/` (Samsung
SM-G998U, 10 x 7 inner corners): fx = 2767, fy = 2775, cx = 1478, cy = 2051 at 3000 x 4000. A calibration
saved from the HW2 page takes priority if one exists.

A 30 s clip takes a minute or two to process at 480 px. That's longer than gunicorn's default 30 s timeout, so
either run it locally and save the result as a sample (below), or start the hosted app with
`gunicorn app:app --timeout 300`. -->

## Sample outputs

Each homework has a **Sample outputs** page showing saved real runs.


## Setup

```bash
python -m venv CVenv
source CVenv/bin/activate
pip install -r requirements.txt
python app.py    # http://localhost:5000
```
