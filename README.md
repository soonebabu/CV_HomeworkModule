# Computer Vision Homeworks (CSc 8830)

Hosted on : https://cv-homeworkmodule-1.onrender.com/

This is one small Flask app that holds all my homework for the course. Every homework has its own page in
the app: each homework also has a **Sample outputs**
page with real runs I saved earlier.




## Structure

```
app.py                           # creates the Flask app and plugs in each homework
homeworks/
  samples.py                     # the "Sample outputs" gallery shared by all homeworks
  hw2_calibration/               # Homework 2 - camera calibration & perspective 
  hw3_blurring/                  # Homework 3 - blurring: spatial filter vs. Fourier domain
    
  hw4_segmentation/               # Homework 4 - classical human boundary segmentation vs. SAM2
  hw6_motion/                     # Homework 6 - optical flow, Lucas-Kanade tracking, structure from motion
  hw7_stereo/                     # Homework 7 - object size with calibrated stereo, room dimensions from video
  challenge_video/                # Challenge 1 - Scene-Graph Orchestrator for consistent long-form video
    notebooks/                    #   Colab notebook (real video model), saved with its outputs
    challenge1_results/           #   outputs of that Colab run (videos, images, scores)
templates/                       # HTML pages, one folder per homework
static/                          # uploads, generated outputs and saved samples
calibration_images/              # chessboard photos used for HW2
object_images/                   # photos of the objects measured in HW2
actualvssystemvalue.csv          # HW2 ground truth vs. what the app measured

```

### Challenge 1

- **Code:** `homeworks/challenge_video/`
- **Notebook:** `homeworks/challenge_video/notebooks/challenge1_colab.ipynb`. It was run on Colab and saved with
  its outputs, so open it on GitHub to see the results.
- **Results page:** `/challenge`, from the Challenge Video Results tile on the landing page.
- **Outputs:** `homeworks/challenge_video/challenge1_results/`, with the side-by-side video (`side_by_side.mp4`),
  each scene's clip (`baseline/`, `orchestrated/`), storyboards, the score chart (`scores.png`) and all numbers
  (`results.json`).



## Run

```bash
python -m venv CVenv
source CVenv/bin/activate
pip install -r requirements.txt
python app.py    # http://localhost:5000
```

Open http://localhost:5000 and pick a homework from the landing page. Each homework has an **Overview** tab
that sums up the task, a tab for every part of the assignment, and a **Sample outputs** tab.


## Homework 2 - Camera Calibration & Measurement

Calibrates a camera from chessboard photos, then uses the intrinsics plus a known camera-to-object distance to
turn a pixel measurement into a real-world length. Validated against a 20+ object ground-truth CSV.

**Files:** `calibration.py`, `measure_dimensions.py`, `validate_measurements.py`, `routes.py`


### Step 1 - Calibrate (`/hw2/calibrate`)

The code finds the chessboard corners in each photo, refines them to sub-pixel accuracy, and runs
`cv2.calibrateCamera`. Photos where the board can't be found are skipped.



### Step 2 - Measure an object (`/hw2/measure`)

The two clicked points are undistorted first with the calibration from step 1. After that it's just the
pinhole camera formula:

```
real length = (pixel length x distance) / focal length
```

x and y are handled separately with fx and fy, then combined. 

### Step 3 - Validate (`/hw2/validate`)


## Homework 3 - Image Blurring: Spatial Filter vs. Fourier Domain

Blurs an uploaded image two independent ways on the same kernel:
- **Spatial**: direct 2D convolution 
- **Frequency**: zero-pad the image and kernel, multiply their FFTs, inverse-FFT, crop back (the standard
  "linear convolution via padded circular convolution" trick, needed because a bare FFT multiply gives circular,
  not linear, convolution).

**Files:** `blur.py`, `routes.py`

### How to use it (`/hw3/blur`)

The image is turned into grayscale and blurred twice with the same kernel:

- **Spatial:** the kernel is flipped (so it's true convolution, not correlation) and applied with
  `cv2.filter2D`, with zeros outside the image.
- **Frequency:** the image is padded with zeros by the kernel's radius, the kernel is placed at the array
  origin with `np.roll`, both go through `np.fft.fft2`, get multiplied, go back through the inverse FFT, and
  the padding is cropped off.

The padding is the important bit. A plain FFT multiply wraps around the edges (circular convolution), so
without it the borders would not match.

## Homework 4 - Human Boundary Segmentation (Classical CV) vs. SAM2

Two classical (non-ML, non-DL) OpenCV segmentation pipelines, each with a boundary/contour output:
**Files:** `rgb_segmentation.py`, `thermal_segmentation.py`, `metrics.py`, `routes.py`

The goal is to cut a person out of a photo and draw their outline using only classical methods

### RGB photos (`/hw4/rgb`)

GrabCut models the colours inside and outside the box and keeps refining which pixels belong to the person.
After that, a morphological open and close cleans up specks and small holes, only the largest connected blob
is kept, and its outer contour becomes the boundary.

### Thermal images (`/hw4/thermal`)
In a thermal image the person is usually the warmest thing in the scene, so this pipeline is much simpler. It
converts to grayscale, blurs lightly, and lets Otsu's method pick the threshold between warm and cold
automatically.


### Comparing with SAM2

SAM2 doesn't run inside this app. Get a mask for your photo from SAM2 separately (the online demo works fine),
save it as an image, and attach it on either page. It gets resized to match if needed, and the page adds:

- **IoU** and **Dice** scores between my mask and the SAM2 mask (1.0 means a perfect match)
- an overlay where **white** is where both agree, **green** is only in my mask, and **red** is only in SAM2's


## Homework 6 - Optical Flow, Tracking and Structure from Motion


**Files:** `optical_flow.py`, `lucas_kanade.py`, `sfm_planar.py`, `video_io.py`, `routes.py`

This one is about motion. There are three parts: watching how everything moves in a video, following
individual points from one frame to the next, and working out the 3D shape of a flat object from four photos
taken at different angles. 

### Part 1 - Optical flow video (`/hw6/flow`)

For every pair of frames, dense Farneback flow gives a motion vector at every pixel. On top of that, a global
affine motion is fitted to each frame, which captures what the camera itself is doing (panning, zooming,
rolling). Anything that doesn't fit that camera motion is something moving on its own, like a person walking
or a car.

### Part 2 - Two-frame tracking (`/hw6/tracking`)

You can get here straight from a finished flow run, and it will reuse the same video. You can also upload a
video directly. Pick a time and the page takes two consecutive frames from that moment.


### Part 3 - Structure from motion (`/hw6/sfm`)

The default camera settings come from the HW2 calibration photos in `calibration_images/` (Samsung
SM-G998U, 10 x 7 inner corners): fx = 2767, fy = 2775, cx = 1478, cy = 2051 at 3000 x 4000. If you've run a
calibration on the HW2 page, that one is used instead.


## Homework 7 - Stereo Measurement and Room Dimensions

**Files:** `stereo.py`, `room.py`, `routes.py`

### Step 1 - Camera intrinsics (`/hw7/intrinsics`)

Click the button. It finds fx, fy, cx, cy and the lens distortion from the chessboard photos in
`calibration_images/`. 
### Step 2 - Object size (`/hw7/stereo`)


The pair is rectified and the main object is found in both images. Its distance comes from Z = f B / d, and you
get width and height for a rectangle, the diameter for a circle, or every edge for any other shape, along with
X, Y and Z for each point. 
### Step 3 - Room dimensions (`/hw7/room`)

