# SPEC3 Color Space

SPEC3 is a new color space that represents colors using a light spectrum compactly represented as 3 values (SX, SY, and SZ) that specifies 3 spectral radiances at the wavelengths 440, 545, and 630nm.

The three values are combined with two bounding points to form a 5-point spectrum:

* *ɑ* at 345 nm (implicitly always = 0)
* **SX** at 440 nm
* **SY** at 545 nm
* **SZ** at 630 nm
* *ω* at 750 nm (implicitly always = 0)

Individual radiances can be computed at any wavelength 
(within the bounds of 345nm to 750nm)
using linear interpolation. This means that the color space can represent any color in the visible spectrum along with
any UV and IR "colors" in a 405nm window by simply tracking a wavelength offset.

Because teh color space of SPEC3 directly measure energy,
addition of colors is simple addition. Because the energies
never need to be clipped, the color space is suitable for
high dynamic range rendering.

The color space is designed to be linearly convertible from CIE XYZ and RGB color spaces with minimal loss of information. Conversion is a simple linear transform so it's fast to convert to and from SPEC3.

## Conversion Functions

A nice feature of SPEC3 is that it's linearly convertible from CIE XYZ and RGB color spaces. Here are the conversion functions:

```python
def xyz_to_spec3(x, y, z):
    sx = 0.001068906393 * x - 0.001293589012 * y + 0.009824186563 * z
    sy = -0.020828193054 * x + 0.025339441374 * y + 0.002468480263 * z
    sz = 0.015242096968 * x - 0.001228092005 * y - 0.002399334684 * z
    sw = 0.007880528457 * x - 0.000641172985 * y - 0.001242348459 * z
    return sx, sy, sz, sw
def spec3_to_xyz(sx, sy, sz, sw):
    x = 16.872589111328 * sx + 4.190485000610 * sy + 55.227638244629 * sz + 27.271245956421 * sw
    y = 3.361047744751 * sx + 42.426795959473 * sy + 52.328025817871 * sz + 10.528821945190 * sw
    z = 100.273048400879 * sx + 5.099486351013 * sy + 0.027020234615 * sz + 0.000008866342 * sw
    return x, y, z

def rgb_to_spec3(r, g, b):
    sx = 0.001750972471 * r + 0.000011275061 * g + 0.009202445857 * b
    sy = 0.000915463548 * r + 0.013870128430 * g + 0.004963375628 * b
    sz = 0.005414634012 * r + 0.002190136351 * g - 0.002131786896 * b
    sw = 0.002796940040 * r + 0.001127772150 * g - 0.001104670228 * b
    return sx, sy, sz, sw
def spec3_to_rgb(sx, sy, sz, sw):
    r = 56.996646881104 * sx - 27.260038375854 * sy + 128.244384765625 * sz + 78.166015625000 * sw
    g = -40.088443756104 * sx + 72.111343383789 * sy + 13.269886016846 * sz - 22.167507171631 * sw
    b = 97.739341735840 * sx + 5.065307617188 * sy - 25.329607009888 * sz - 13.158031463623 * sw
    return r, g, b
```

```javascript
xyz_to_spec3(xyz: vec3<f32>) -> vec4<f32> {
    const xyz_to_spec3: mat4x3<f32> = mat4x3(
        0.001068906393, -0.001293589012, 0.009824186563, 
        -0.020828193054, 0.025339441374, 0.002468480263, 
        0.015242096968, -0.001228092005, -0.002399334684, 
        0.007880528457, -0.000641172985, -0.001242348459, 
    );
    return xyz_to_spec3 * xyz;
}
spec3_to_xyz(spec3: vec4<f32>) -> vec3<f32> {
    const spec3_to_xyz: mat3x4<f32> = mat3x4(
        16.872589111328, 4.190485000610, 55.227638244629, 27.271245956421, 
        3.361047744751, 42.426795959473, 52.328025817871, 10.528821945190, 
        100.273048400879, 5.099486351013, 0.027020234615, 0.000008866342, 
    );
    return spec3_to_xyz * spec3;
}

rgb_to_spec3(rgb: vec3<f32>) -> vec4<f32> {
    const rgb_to_spec3: mat4x3<f32> = mat4x3(
        0.001750972471, 0.000011275061, 0.009202445857, 
        0.000915463548, 0.013870128430, 0.004963375628, 
        0.005414634012, 0.002190136351, -0.002131786896, 
        0.002796940040, 0.001127772150, -0.001104670228, 
    );
    return rgb_to_spec3 * rgb;
}
spec3_to_rgb(spec3: vec4<f32>) -> vec3<f32> {
    const spec3_to_rgb: mat3x4<f32> = mat3x4(
        56.996646881104, -27.260038375854, 128.244384765625, 78.166015625000, 
        -40.088443756104, 72.111343383789, 13.269886016846, -22.167507171631, 
        97.739341735840, 5.065307617188, -25.329607009888, -13.158031463623, 
    );
    return spec3_to_rgb * spec3;
}
```


## Why Another Color Space?

Benefits:

- [x] Compact representation suitable for realtime GPU rendering (SX, SY, SZ, SW happily fit into 16 or 32-bit vectors)
- [x] High dynamic range since radiance values can be accumulated without clipping.

