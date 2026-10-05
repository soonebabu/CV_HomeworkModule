# Computer Vision Homeworks (CSc 8830)

Hosted on : https://cv-homeworkmodule-1.onrender.com/

This is one small Flask app that holds all my homework for the course. Every homework has its own page in
the app: you upload your own photo or video, the code runs on it, and you see the result straight away along
with the numbers behind it. If you'd rather not upload anything, each homework also has a **Sample outputs**
page with real runs I saved earlier.

The sections below go through each module: what it does, how to use it in the app, and what you should see
when it works.


## Structure

```
app.py                           # creates the Flask app and plugs in each homework
homeworks/
  samples.py                     # the "Sample outputs" gallery shared by all homeworks
  hw2_calibration/               # Homework 2 - camera calibration & perspective 
  hw3_blurring/                  # Homework 3 - blurring: spatial filter vs. Fourier domain
    
  hw4_segmentation/               # Homework 4 - classical human boundary segmentation vs. SAM2
  hw6_motion/                     # Homework 6 - optical flow, Lucas-Kanade tracking, structure from motion
templates/                       # HTML pages, one folder per homework
static/                          # uploads, generated outputs and saved samples
calibration_images/              # chessboard photos used for HW2
object_images/                   # photos of the objects measured in HW2
actualvssystemvalue.csv          # HW2 ground truth vs. what the app measured

```


## Getting it running

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

The idea is simple. If you know how far away an object is and you know your camera's focal length in pixels,
you can turn a length in pixels into a length in millimetres. The hard part is getting a trustworthy focal
length, and that's what calibration is for. The homework has three steps, and you do them in order.

### Step 1 - Calibrate (`/hw2/calibrate`)

1. Print a chessboard and take 10 to 15 photos of it from different angles and distances. Use the same phone
   and the same zoom you'll use for measuring later. The photos I used are in `calibration_images/`.
2. Upload them all at once and enter the board size. Count the **inner corners** (where four squares meet),
   not the squares. My board has 10 x 7 inner corners. Also enter the size of one square in mm.
3. Click calibrate.

The code finds the chessboard corners in each photo, refines them to sub-pixel accuracy, and runs
`cv2.calibrateCamera`. Photos where the board can't be found are skipped.

**What you should see:** fx, fy (focal length in pixels), cx, cy (the optical centre), the five distortion
coefficients, how many of your photos were actually used, and the mean reprojection error. An error below
about 1 px means the calibration is good. The result is saved to `static/calib/camera_calib.npz`, so the
other pages (and HW6) can use it.

### Step 2 - Measure an object (`/hw2/measure`)

1. Take a photo of an object from a known distance. Measure that distance with a tape and write it down.
2. Upload the photo, then click the two ends of the edge you want to measure.
3. Enter the camera-to-object distance in mm and hit compute.

The two clicked points are undistorted first with the calibration from step 1. After that it's just the
pinhole camera formula:

```
real length = (pixel length x distance) / focal length
```

x and y are handled separately with fx and fy, then combined. **What you should see:** the pixel distance
between your clicks and the real length in mm and cm. If you go to this page before calibrating, it will send
you back to the calibration page.

### Step 3 - Validate (`/hw2/validate`)

Upload a CSV with the columns `object_id, actual_mm, measured_mm, distance_mm`. The one I submitted is
`actualvssystemvalue.csv`: 21 objects measured with a ruler and then with the app, all shot from 2.2 m.

**What you should see:** a table with the error for each object, summary numbers (mean absolute error, RMSE,
standard deviation, mean % error, max error), and two charts. One puts actual and measured side by side for
every object, the other shows the percent error per object.

### Running it without the web app

All three scripts also work from the terminal:

```bash
python homeworks/hw2_calibration/calibration.py --images calibration_images --cols 10 --rows 7 --square <mm>
python homeworks/hw2_calibration/measure_dimensions.py --image object_images/object1.jpg --distance 2200
python homeworks/hw2_calibration/validate_measurements.py --csv actualvssystemvalue.csv
```

If you leave out `--p1` and `--p2`, `measure_dimensions.py` opens a window so you can click the two points.


## Homework 3 - Image Blurring: Spatial Filter vs. Fourier Domain

Blurs an uploaded image two independent ways on the same kernel:
- **Spatial**: direct 2D convolution 
- **Frequency**: zero-pad the image and kernel, multiply their FFTs, inverse-FFT, crop back (the standard
  "linear convolution via padded circular convolution" trick, needed because a bare FFT multiply gives circular,
  not linear, convolution).

**Files:** `blur.py`, `routes.py`

This homework tests the convolution theorem: blurring an image in the spatial domain should give exactly the
same result as multiplying in the frequency domain. If both code paths are right, the two blurred images
should match down to floating-point rounding.

### How to use it (`/hw3/blur`)

1. Upload any photo. Big phone photos get shrunk to 900 px on the long side so the page stays quick.
2. Pick a kernel: **Gaussian** (set the size and sigma) or **Box** (just the size). Even sizes are bumped up
   to the next odd number so the kernel has a centre pixel.
3. Hit run.

The image is turned into grayscale and blurred twice with the same kernel:

- **Spatial:** the kernel is flipped (so it's true convolution, not correlation) and applied with
  `cv2.filter2D`, with zeros outside the image.
- **Frequency:** the image is padded with zeros by the kernel's radius, the kernel is placed at the array
  origin with `np.roll`, both go through `np.fft.fft2`, get multiplied, go back through the inverse FFT, and
  the padding is cropped off.

The padding is the important bit. A plain FFT multiply wraps around the edges (circular convolution), so
without it the borders would not match.

**What you should see:** one 2 x 3 figure. The top row has the original image, the kernel, and the spatial
result. The bottom row has the image spectrum, the kernel's frequency response, and the frequency-domain
result. Under that there's a difference image and the numbers comparing the two results: max and mean
absolute difference, MSE and PSNR. On a working run the differences are tiny (around 1e-12), and PSNR comes
out as infinite or extremely high. That's the convolution theorem holding.

The **Theory** tab walks through the maths behind all of this.


## Homework 4 - Human Boundary Segmentation (Classical CV) vs. SAM2

Two classical (non-ML, non-DL) OpenCV segmentation pipelines, each with a boundary/contour output:
- **RGB**: GrabCut (iterative graph-cut / GMM energy minimization on a single image, no trained model) seeded
  with a rectangle you drag on the photo, or an automatic whole-image box.
- **Thermal**: Otsu global thresholding + morphology - a person is almost always the warmest (brightest) blob in
  a thermal frame.

**Files:** `rgb_segmentation.py`, `thermal_segmentation.py`, `metrics.py`, `routes.py`

The goal is to cut a person out of a photo and draw their outline using only classical methods: no trained
model, no weights, no dataset. The results are then compared against a mask from SAM2, a modern deep
learning model, to see how close the classical approach gets.

### RGB photos (`/hw4/rgb`)

1. Upload a photo of a person.
2. Drag a rough box around them. It doesn't need to be tight. If you don't want to draw one, tick the
   automatic option and the whole image (inset by 5%) is used.
3. Optionally attach a SAM2 mask of the same photo (more on this below), then run.

GrabCut models the colours inside and outside the box and keeps refining which pixels belong to the person.
After that, a morphological open and close cleans up specks and small holes, only the largest connected blob
is kept, and its outer contour becomes the boundary.

**What you should see:** the box you drew, the black and white mask, and the photo with the person's outline
drawn in green.

### Thermal images (`/hw4/thermal`)

1. Upload a thermal image. Raw grayscale frames and false-colour images (ironbow, white-hot) both work.
2. Leave "hot is bright" ticked for normal palettes. Untick it for black-hot cameras, where warm things show
   up dark.
3. Optionally attach a SAM2 mask, then run.

In a thermal image the person is usually the warmest thing in the scene, so this pipeline is much simpler. It
converts to grayscale, blurs lightly, and lets Otsu's method pick the threshold between warm and cold
automatically. The same cleanup and largest-blob step follows.

**What you should see:** the input, the grayscale image the threshold was applied to, the mask, and the
outline drawn in orange.

### Comparing with SAM2

SAM2 doesn't run inside this app. Get a mask for your photo from SAM2 separately (the online demo works fine),
save it as an image, and attach it on either page. It gets resized to match if needed, and the page adds:

- **IoU** and **Dice** scores between my mask and the SAM2 mask (1.0 means a perfect match)
- an overlay where **white** is where both agree, **green** is only in my mask, and **red** is only in SAM2's

The **Theory** tab explains how GrabCut and Otsu work.



<!-- ## Homework 6 - Optical Flow, Tracking and Structure from Motion -->

<!--
**Files:** `optical_flow.py`, `lucas_kanade.py`, `sfm_planar.py`, `video_io.py`, `routes.py`

This one is about motion. There are three parts: watching how everything moves in a video, following
individual points from one frame to the next, and working out the 3D shape of a flat object from four photos
taken at different angles. There's also a **Theory** tab with all the derivations. The code follows those
derivations closely, so it's worth reading the two together.

### Part 1 - Optical flow video (`/hw6/flow`)

1. Upload a phone video (MP4 or MOV straight off the phone works best).
2. Choose where the clip starts and how long it is. 30 s is the default and 60 s is the limit.
3. Optionally change the processing size (480 px by default), the frame step, and the motion threshold
   (1 px per frame by default, anything slower counts as still). Then run.

For every pair of frames, dense Farneback flow gives a motion vector at every pixel. On top of that, a global
affine motion is fitted to each frame, which captures what the camera itself is doing (panning, zooming,
rolling). Anything that doesn't fit that camera motion is something moving on its own, like a person walking
or a car.

What you should see:

- a **flow video**: the original frames with arrows on the left, and colour-coded flow on the right (the
  colour shows the direction and the brightness shows the speed; black means nothing is moving)
- a **timeline** of mean speed, how much of the frame is moving, and the camera pan over the clip
- the **busiest frame** shown three ways: arrows, colour flow, and the independently moving pixels in red
- two **heat maps**: where motion happened overall, and where it happened once camera motion is removed
- a **direction histogram** showing which way things mostly moved
- a short list of **observations** in plain words, each one backed by a number from the run
- a CSV with the stats for every frame

A 30 s clip takes a minute or two to process at 480 px. That's longer than gunicorn's default 30 s timeout, so
either run it locally and save the result as a sample (see below), or start the hosted app with
`gunicorn app:app --timeout 300`.

### Part 2 - Two-frame tracking (`/hw6/tracking`)

You can get here straight from a finished flow run, and it will reuse the same video. You can also upload a
video directly. Pick a time and the page takes two consecutive frames from that moment.

1. Choose how many features to track (40 by default).
2. Optionally click up to 10 points by hand in both frames, to check against later.
3. Run.

About 40 good corners (Shi-Tomasi) are picked, preferring ones on things that actually move. Each one is then
tracked with a pyramidal Lucas-Kanade tracker I wrote from the equations in `lucas_kanade.py`, bilinear
interpolation included. It uses a 21 x 21 window and 4 pyramid levels, and keeps iterating until the update
drops below 0.01 px. Every prediction is checked against three things:

- the **actual** position, found separately by NCC template matching with sub-pixel peak fitting (matches
  weaker than 0.9 aren't trusted as ground truth)
- the points you **clicked** by hand
- OpenCV's own `calcOpticalFlowPyrLK` on the same points

What you should see: both frames with the tracks drawn on them, a difference image showing where the frames
changed, 10x zoomed crops of the biggest movers so you can see the actual pixels, a predicted vs. actual plot,
and a table for every feature. The headline numbers are the median error against the actual position, the
share of points within 0.5 px and within 1 px, and how close my tracker gets to OpenCV's. One feature is also
worked through iteration by iteration, with every number shown, so you can follow the maths by hand.

### Part 3 - Structure from motion (`/hw6/sfm`)

1. Take four photos of a flat object (a book, a card, a sheet of paper) from four different angles.
2. Upload all four. The camera settings are filled in for you (see the note below), and you can change them.
3. Click the same points on the object in each photo, in the same order: corners first, then a few points
   along the edges. You need at least 4, and 6 to 10 works best.
4. Tell it which two points have a known distance between them and what that distance is in cm. This is what
   turns the result into real units. If you have actual lengths for the edges, you can enter those too, and
   they'll be compared against what was recovered.

Because the object is flat, the usual essential matrix approach breaks down, so this goes through
homographies instead:

- a normalised DLT homography from view 1 to each of views 2, 3 and 4
- each one decomposed into a rotation R, a translation t/d, and the plane normal N. There are four possible
  answers each time, and the right one is picked by keeping points in front of the camera and making sure all
  three views agree on the same normal.
- every point triangulated from all four views, then refined together with bundle adjustment
- a plane fitted through the points and everything scaled to cm using your known length

What you should see: a 3D plot of the object with the four camera positions, the recovered outline of the
object with edge lengths, area and perimeter, a top-down view of the object made from view 1, and each photo
with your clicks (red) next to the reprojected points (green). If those red and green dots sit on top of each
other, the reconstruction is consistent. All the intermediate matrices are printed on the page, and there's a
JSON report you can download with the camera parameters, EXIF data, clicks and poses.

The default camera settings come from the HW2 calibration photos in `calibration_images/` (Samsung
SM-G998U, 10 x 7 inner corners): fx = 2767, fy = 2775, cx = 1478, cy = 2051 at 3000 x 4000. If you've run a
calibration on the HW2 page, that one is used instead.
-->


## Sample outputs

Each homework has a **Sample outputs** page showing saved real runs, so anyone can see what the results look
like without uploading anything.

Here's how saving works. Whenever you run something locally, a **Save as sample** button shows up under the
result. Clicking it copies the images and numbers from that run into `static/samples/<homework>/`. Commit that
folder and push, and the sample appears on the hosted site too. Render wipes its disk on every deploy, so
saving there wouldn't last. That's why the button only shows up when the app runs locally in debug mode, or
when `SAMPLES_EDITABLE=1` is set.
