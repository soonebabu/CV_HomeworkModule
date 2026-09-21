"""
Image blurring via spatial filtering and its Fourier-domain equivalent.

The core claim under test (the convolution theorem):

    f(x,y) * h(x,y)  <-->  F(u,v) . H(u,v)

i.e. convolving an image with a kernel in the spatial domain gives the exact
same result as multiplying their Fourier transforms and taking the inverse
transform. ``blur_compare`` computes the blur both ways on the same
(grayscale) image and returns both results plus the numerical difference
between them, so the equivalence can be checked directly rather than just
asserted. ``build_comparison_figure`` lays the whole experiment out as one
2x3 figure: original / kernel / spatial result on top, image spectrum /
kernel spectrum (OTF) / frequency-domain result on the bottom.
"""

import cv2
import matplotlib
matplotlib.use("Agg")  # headless-safe, works from the web app too
import matplotlib.pyplot as plt
import numpy as np


def gaussian_kernel_2d(ksize, sigma):
    if ksize % 2 == 0:
        ksize += 1
    ax = np.arange(-(ksize // 2), ksize // 2 + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def box_kernel_2d(ksize):
    if ksize % 2 == 0:
        ksize += 1
    kernel = np.ones((ksize, ksize), dtype=np.float64)
    kernel /= kernel.sum()
    return kernel


def spatial_convolve(channel, kernel):
    """True convolution (not correlation) via cv2.filter2D, zero-padded border."""
    flipped = kernel[::-1, ::-1]
    return cv2.filter2D(channel.astype(np.float64), -1, flipped, borderType=cv2.BORDER_CONSTANT)


def frequency_convolve(channel, kernel):
    """
    Linear convolution computed as a padded circular convolution in the
    Fourier domain, matching spatial_convolve's zero-padded border exactly.

    A plain FFT multiply on the original image size gives *circular*
    convolution (edges wrap around). To get the same result as the
    zero-padded spatial convolution, the image is first zero-padded by the
    kernel's radius on every side; that padding gives the wraparound enough
    "room" to land entirely inside the pad, so the center region of the
    result equals true linear convolution. The kernel is placed at the
    array origin (np.roll) since FFT-based convolution assumes the kernel
    is centered at index (0, 0), not at its visual center.
    """
    h, w = channel.shape
    kh, kw = kernel.shape
    pad_y, pad_x = kh // 2, kw // 2

    padded = cv2.copyMakeBorder(channel.astype(np.float64), pad_y, pad_y, pad_x, pad_x,
                                 cv2.BORDER_CONSTANT, value=0)
    ph, pw = padded.shape

    kernel_full = np.zeros((ph, pw), dtype=np.float64)
    kernel_full[:kh, :kw] = kernel
    kernel_full = np.roll(kernel_full, (-pad_y, -pad_x), axis=(0, 1))

    F_img = np.fft.fft2(padded)
    F_kernel = np.fft.fft2(kernel_full)
    full_result = np.fft.ifft2(F_img * F_kernel).real

    return full_result[pad_y:pad_y + h, pad_x:pad_x + w]


def magnitude_spectrum(channel):
    """Log-scaled, fftshifted magnitude spectrum (float, not yet normalized for display)."""
    F = np.fft.fftshift(np.fft.fft2(channel.astype(np.float64)))
    return np.log1p(np.abs(F))


def kernel_frequency_response(kernel, shape):
    """Magnitude of the kernel's own frequency response (OTF), padded to `shape`, fftshifted."""
    h, w = shape
    kh, kw = kernel.shape
    padded = np.zeros((h, w), dtype=np.float64)
    padded[:kh, :kw] = kernel
    padded = np.roll(padded, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    F = np.fft.fftshift(np.fft.fft2(padded))
    return np.abs(F)


def normalize_for_display(arr):
    a = arr.astype(np.float64)
    if a.max() > 1e-12:
        a = (a / a.max()) * 255.0
    return np.clip(a, 0, 255).astype(np.uint8)


def blur_compare(img_bgr, kernel_type="gaussian", ksize=15, sigma=3.0):
    """
    Runs spatial-domain and frequency-domain blurring on the same grayscale
    image with the same kernel and returns everything needed to display and
    validate the comparison.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if kernel_type == "box":
        kernel = box_kernel_2d(ksize)
    else:
        kernel = gaussian_kernel_2d(ksize, sigma)

    spatial_result = spatial_convolve(gray, kernel)
    freq_result = frequency_convolve(gray, kernel)
    diff = spatial_result - freq_result

    metrics = {
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "mse": float(np.mean(diff ** 2)),
        "psnr_db": _psnr(spatial_result, freq_result),
    }

    return {
        "gray": gray,
        "kernel": kernel,
        "spatial_u8": np.clip(spatial_result, 0, 255).astype(np.uint8),
        "freq_u8": np.clip(freq_result, 0, 255).astype(np.uint8),
        "diff_vis": normalize_for_display(np.abs(diff)),
        "metrics": metrics,
        "image_spectrum": magnitude_spectrum(gray),
        "kernel_spectrum": kernel_frequency_response(kernel, gray.shape),
    }


def build_comparison_figure(result, kernel_type, out_path):
    """
    Lays the whole experiment out as one 2x3 figure, matching the classic
    "spatial vs. frequency" textbook comparison:

        Original Image   | {Kernel} Kernel    | Spatial Convolution
        Image Spectrum    | Kernel Spectrum    | Frequency-Domain Filtering
        (FFT)              (OTF)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.patch.set_facecolor("white")

    kernel_title = f"{kernel_type.capitalize()} Kernel"
    panels = [
        (axes[0, 0], result["gray"], "gray", "Original Image"),
        (axes[0, 1], result["kernel"], "hot", kernel_title),
        (axes[0, 2], result["spatial_u8"], "gray", "Spatial Convolution"),
        (axes[1, 0], result["image_spectrum"], "gray", "Image Spectrum (FFT)"),
        (axes[1, 1], result["kernel_spectrum"], "gray", "Kernel Spectrum (OTF)"),
        (axes[1, 2], result["freq_u8"], "gray", "Frequency-Domain Filtering"),
    ]
    for ax, data, cmap, title in panels:
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=13)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)


def _psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return None
    return 10.0 * np.log10((255.0 ** 2) / mse)
