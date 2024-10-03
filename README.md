# SPEC4 Color Format

SPEC4 is a new color format that represents colors using a
4-point spectrogram in the human perceptual color space.

Each of the four points represents the spectral radiance
(watts per steradian per square meter per meter, don't worry about it) of a bandwidth of light.

The four points are:

* *ɑ* at 400 nm (implicit always = 0)
* **X** at 460 nm
* **Y** at 520 nm
* **Z** at 580 nm
* **W** at 640 nm
* *ω* at 700 nm (implicit always = 0)

The values X, Y, Z, and W are the radiance of the color at the respective wavelengths but are not computed to be impulses at those wavelengths. Instead, they are computed to be the average radiance of the color over the bandwidth of the respective wavelength. If you're not into color theory, again, don't worry about it - I just want to be clear. For details see `fit.py`.


## Why Another Color Space?

Benefits:

- [x] Compact representation suitable for realtime GPU rendering (X, Y, Z, W happily fit into 16 or 32-bit vectors)
- [x] High dynamic range since radiance values can be accumulated without clipping.

