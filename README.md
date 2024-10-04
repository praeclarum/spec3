# SPEC4 Color Format

SPEC4 is a new color format that represents colors using a
4-point spectrogram in the human perceptual color space.

Each of the four points represents the spectral radiance
(watts per steradian per square meter per nanometer, don't worry about it) of a bandwidth of light.

The four points are:

* *ɑ* at 400 nm (implicitly always = 0)
* **SX** at 460 nm
* **SY** at 520 nm
* **SZ** at 580 nm
* **SW** at 640 nm
* *ω* at 700 nm (implicitly always = 0)

The values SX, SY, SZ, and SW are the radiance of the color at the respective wavelengths but are not computed to be impulses at those wavelengths. Instead, they are computed to be the average radiance of the color over the bandwidth of the respective wavelength. If you're not into color theory, again, don't worry about it - I just want to be clear. For details see `fit.py`.


## Conversion Functions

A nice feature of SPEC4 is that it's linearly convertible from CIE XYZ and RGB color spaces. Here are the conversion functions:

```python
def xyz_to_spec4(x, y, z):
    sx = 0.001068906393 * x + -0.001293589012 * y + 0.009824186563 * z
    sy = -0.020828193054 * x + 0.025339441374 * y + 0.002468480263 * z
    sz = 0.015242096968 * x + -0.001228092005 * y + -0.002399334684 * z
    sw = 0.007880528457 * x + -0.000641172985 * y + -0.001242348459 * z
    return sx, sy, sz, sw

def spec4_to_xyz(sx, sy, sz, sw):
    x = 16.872589111328 * sx + 4.190485000610 * sy + 55.227638244629 * sz + 27.271245956421 * sw
    y = 3.361047744751 * sx + 42.426795959473 * sy + 52.328025817871 * sz + 10.528821945190 * sw
    z = 100.273048400879 * sx + 5.099486351013 * sy + 0.027020234615 * sz + 0.000008866342 * sw
    return x, y, z

def rgb_to_spec4(r, g, b):
    sx = 0.001750972471 * r + 0.000011275061 * g + 0.009202445857 * b
    sy = 0.000915463548 * r + 0.013870128430 * g + 0.004963375628 * b
    sz = 0.005414634012 * r + 0.002190136351 * g + -0.002131786896 * b
    sw = 0.002796940040 * r + 0.001127772150 * g + -0.001104670228 * b
    return sx, sy, sz, sw
```


## Why Another Color Space?

Benefits:

- [x] Compact representation suitable for realtime GPU rendering (SX, SY, SZ, SW happily fit into 16 or 32-bit vectors)
- [x] High dynamic range since radiance values can be accumulated without clipping.

