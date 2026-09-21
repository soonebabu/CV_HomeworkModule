# Computer Vision Homeworks (CSc 8830)

Hosted on : https://cv-homeworkmodule-1.onrender.com/


## Structure

```
app.py                          
homeworks/
  hw2_calibration/               # Homework 2 - camera calibration & perspective 
  hw3_blurring/                  # Homework 3 - blurring: spatial filter vs. Fourier domain
    
  hw4_segmentation/               # Homework 4 - classical human boundary segmentation vs. SAM2
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



## Setup

```bash
python -m venv CVenv
source CVenv/bin/activate
pip install -r requirements.txt
python app.py    # http://localhost:5000
```
