"""Collection of functions to convert between different color spaces."""

import torch
from torch import Tensor

#
# Primary color space conversion functions
#

RGB_to_XYZ_matrix = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
])

XYZ_to_RGB_matrix = torch.inverse(RGB_to_XYZ_matrix)

def batched_sRGB_to_RGB(srgb: Tensor) -> Tensor:
    """Converts sRGB to linear RGB using a gamma of 2.4.

    Args:
        srgb: A tensor of sRGB values in the range [0, 1]
              shaped as (batch_size, 3).

    Returns:
        A tensor of linear RGB values shaped as (batch_size, 3).
    """
    return torch.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)

def batched_RGB_to_sRGB(rgb: Tensor) -> Tensor:
    """Converts linear RGB to sRGB using a gamma of 2.4 and clips the result to [0, 1].

    Args:
        rgb: A tensor of linear RGB values shaped as (batch_size, 3).

    Returns:
        A tensor of sRGB values in the range [0, 1] shaped as (batch_size, 3).
    """
    unclipped_sRGB = torch.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * (rgb ** (1 / 2.4)) - 0.055)
    srgb = torch.clamp(unclipped_sRGB, 0.0, 1.0)
    return srgb

def batched_RGB_to_XYZ(rgb: Tensor) -> Tensor:
    """Converts linear RGB to CIE XYZ.

    Args:
        rgb: A tensor of linear RGB values shaped as (batch_size, 3).

    Returns:
        A tensor of CIE XYZ values shaped as (batch_size, 3).
    """
    return torch.matmul(rgb, RGB_to_XYZ_matrix)

def batched_XYZ_to_RGB(xyz: Tensor) -> Tensor:
    """Converts CIE XYZ to linear RGB.

    Args:
        xyz: A tensor of CIE XYZ values shaped as (batch_size, 3).

    Returns:
        A tensor of linear RGB values shaped as (batch_size, 3).
    """
    return torch.matmul(xyz, XYZ_to_RGB_matrix)

def tone_map_XYZ(xyz: Tensor) -> Tensor:
    """Tone maps CIE XYZ values using Reinhard's method.

    Args:
        xyz: A tensor of CIE XYZ values shaped as (batch_size, 3).

    Returns:
        A tensor of tone-mapped CIE XYZ values shaped as (batch_size, 3).
    """
    return xyz / (1 + xyz)

def batched_XYZ_to_xyY(xyz: Tensor, eps: float=1e-7) -> Tensor:
    """Converts CIE XYZ to CIE xyY.

    Args:
        xyz: A tensor of CIE XYZ values shaped as (batch_size, 3).
        eps: A small value to avoid division by zero when normalizing.

    Returns:
        A tensor of CIE xyY values shaped as (batch_size, 3).
    """
    sum_xyz = torch.sum(xyz, dim=1, keepdim=True)
    xy = xyz[:, :2] / (sum_xyz + eps)
    Y = xyz[:, 1:2]
    return torch.cat([xy, Y], dim=1)

def batched_xyY_to_XYZ(xyY: Tensor) -> Tensor:
    """Converts CIE xyY to CIE XYZ.

    Args:
        xyY: A tensor of CIE xyY values shaped as (batch_size, 3).

    Returns:
        A tensor of CIE XYZ values shaped as (batch_size, 3).
    """
    x, y, Y = xyY[:, 0], xyY[:, 1], xyY[:, 2]
    scale = Y / y
    X = x * scale
    Z = (1 - x - y) * scale
    return torch.stack([X, Y, Z], dim=1)

spec4_wavelengths = torch.tensor([400.0, 460.0, 520.0, 580.0, 640.0, 700.0])

RGB_to_SPEC4_matrix = torch.tensor([[ 1.7695e-03,  2.9910e-05,  9.2270e-03],
        [ 8.2094e-04,  1.3803e-02,  5.1758e-03],
        [ 5.4744e-03,  2.2374e-03, -2.2369e-03],
        [ 2.8392e-03,  1.1650e-03, -1.1494e-03]]).T

# ERROR: linalg.inv: A must be batches of square matrices, but they are 4 by 3 matrices
# SPEC4_to_RGB_matrix = torch.inverse(RGB_to_SPEC4_matrix)

def batched_RGB_to_SPEC4(rgb: Tensor, clip: bool = True) -> Tensor:
    """Converts linear RGB to SPEC4.

    Args:
        rgb: A tensor of linear RGB values shaped as (batch_size, 3).

    Returns:
        A tensor of SPEC4 values shaped as (batch_size, 4).
    """
    spec4 = torch.matmul(rgb, RGB_to_SPEC4_matrix)
    if clip:
        spec4 = torch.nn.functional.relu(spec4)
    return spec4

def piecewise_gaussian(x: Tensor, mu: float, tau1: float, tau2: float):
    """A piecewise Gaussian function with different slopes on the left and right.

    Args:
        x: The input tensor.
        mu: The mean of the Gaussian.
        tau1: 1/stddev on the negative side.
        tau2: 1/stddev on the positive side.

    Returns:
        A tensor of the same shape as x.
    """
    return torch.where(
        x < mu,
        torch.exp(-tau1**2 * (x - mu)**2 / 2),
        torch.exp(-tau2**2 * (x - mu)**2 / 2))

def xyz_color_matching(wavelength: Tensor) -> Tensor:
    """The x, y, z color matching functions for the CIE 1931 2-degree standard observer.

    Args:
        wavelength: The wavelength in nanometers shaped as (batch_size,).

    Returns:
        The x, y, and z color matching function values shaped as (batch_size, 3).
    """
    x = 1.056 * piecewise_gaussian(wavelength, 599.8, 0.0264, 0.0323) + \
        0.362 * piecewise_gaussian(wavelength, 442.0, 0.0624, 0.0374) - \
        0.065 * piecewise_gaussian(wavelength, 501.1, 0.0490, 0.0382)
    y = 0.821 * piecewise_gaussian(wavelength, 568.8, 0.0213, 0.0247) + \
        0.286 * piecewise_gaussian(wavelength, 530.9, 0.0613, 0.0322)
    z = 1.217 * piecewise_gaussian(wavelength, 437.0, 0.0845, 0.0278) + \
        0.681 * piecewise_gaussian(wavelength, 459.0, 0.0385, 0.0725)
    
    return torch.stack([x, y, z], dim=1)

def batched_spectrum_to_XYZ(spectral_radiance: Tensor, wavelengths: Tensor) -> Tensor:
    """Converts spectral radiance to CIE XYZ.
    Integral(spectral_radiance(wavelength) * xyz_color_matching(wavelength) * dwavelength, wavelength)

    Args:
        spectral_radiance: A tensor of spectral radiance values shaped as (batch_size, n_wavelengths).
        wavelengths: A tensor of discrete wavelengths in nanometers shaped as (n_wavelengths,).

    Returns:
        A tensor of CIE XYZ values shaped as (batch_size, 3).
    """
    color_matching = xyz_color_matching(wavelengths)
    dwavelength = wavelengths[1:] - wavelengths[:-1]
    matched_radiance = spectral_radiance.unsqueeze(-1) * color_matching
    mean_matched_radiance = (matched_radiance[:, :-1, :] + matched_radiance[:, 1:, :]) / 2
    xyz = torch.sum(mean_matched_radiance * dwavelength.unsqueeze(0).unsqueeze(-1), dim=1)
    return xyz

def batched_SPEC4_to_XYZ(spec4: Tensor) -> Tensor:
    """Converts SPEC4 to CIE XYZ.

    Args:
        spec4: A tensor of SPEC4 values shaped as (batch_size, 4).

    Returns:
        A tensor of CIE XYZ values shaped as (batch_size, 3).
    """
    spectrum = torch.nn.functional.pad(spec4, (1, 1), value=0.0)
    xyz = batched_spectrum_to_XYZ(spectrum, spec4_wavelengths)
    return xyz

#
# Composite (mult-step) conversions
#

def batched_sRGB_to_XYZ(srgb):
    """Converts sRGB to CIE XYZ.

    Args:
        srgb: A tensor of sRGB values in the range [0, 1]
              shaped as (batch_size, 3).

    Returns:
        A tensor of CIE XYZ values shaped as (batch_size, 3).
    """
    return batched_RGB_to_XYZ(batched_sRGB_to_RGB(srgb))

def batched_XYZ_to_sRGB(xyz):
    """Converts CIE XYZ to sRGB in the range [0, 1].
    If XYZ is HDR, you should apply tone mapping using `tone_map_XYZ` before converting to sRGB.

    Args:
        xyz: A tensor of CIE XYZ values shaped as (batch_size, 3).

    Returns:
        A tensor of sRGB values in the range [0, 1] shaped as (batch_size, 3).
    """
    return batched_RGB_to_sRGB(batched_XYZ_to_RGB(xyz))

def batched_sRGB_to_SPEC4(srgb):
    """Converts sRGB to SPEC4.

    Args:
        srgb: A tensor of sRGB values in the range [0, 1]
              shaped as (batch_size, 3).

    Returns:
        A tensor of SPEC4 values shaped as (batch_size, 4).
    """
    return batched_RGB_to_SPEC4(batched_sRGB_to_RGB(srgb))

def batched_SPEC4_to_RGB(spec4: Tensor) -> Tensor:
    """Converts SPEC4 to linear RGB.

    Args:
        spec4: A tensor of SPEC4 values shaped as (batch_size, 4).

    Returns:
        A tensor of linear RGB values shaped as (batch_size, 3).
    """
    # Would like to do this, but the matrix is not square
    # return torch.matmul(spec4, SPEC4_to_RGB_matrix)
    xyz = batched_SPEC4_to_XYZ(spec4)
    return batched_XYZ_to_RGB(xyz)

def batched_SPEC4_to_sRGB(spec4):
    """Converts SPEC4 to sRGB in the range [0, 1].

    Args:
        spec4: A tensor of SPEC4 values shaped as (batch_size, 4).

    Returns:
        A tensor of sRGB values in the range [0, 1] shaped as (batch_size, 3).
    """
    return batched_RGB_to_sRGB(batched_SPEC4_to_RGB(spec4))

def batched_XYZ_to_SPEC4(xyz):
    """Converts CIE XYZ to SPEC4.

    Args:
        xyz: A tensor of CIE XYZ values shaped as (batch_size, 3).

    Returns:
        A tensor of SPEC4 values shaped as (batch_size, 4).
    """
    rgb = batched_XYZ_to_RGB(xyz)
    return batched_RGB_to_SPEC4(rgb)

#
# Testing
#

def test_luminance():
    srgb = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.5, 0.5, 0.5],
        [0.1, 0.1, 0.1],
        [0.0, 0.0, 0.0],
    ])
    print(f"srgb: {srgb}")
    rgb = batched_sRGB_to_RGB(srgb)
    print(f"rgb: {rgb}")
    xyz = batched_sRGB_to_XYZ(rgb)
    print(f"xyz: {xyz}")
    xyz2 = 2.0 * xyz
    xyz_luminance = torch.sum(xyz, dim=1, keepdim=False)
    xyz_luminance2 = torch.sum(xyz2, dim=1, keepdim=False)
    print(f"xyz_luminance: {xyz_luminance}")
    print(f"xyz_luminance2: {xyz_luminance2}")
    spec4 = batched_XYZ_to_SPEC4(xyz)
    spec42 = batched_XYZ_to_SPEC4(xyz2)
    print(f"spec4: {spec4}")
    print(f"spec42: {spec42}")
    dwavelength = 60.0
    spec4_luminance = torch.sum(spec4, dim=1) * dwavelength
    spec4_luminance2 = torch.sum(spec42, dim=1) * dwavelength
    print(f"spec4_luminance: {spec4_luminance}")
    print(f"spec4_luminance2: {spec4_luminance2}")

def test_round_trip(title, input_data, conversion_fn, inverse_fn):
    """Tests that the conversion and inverse functions are consistent."""
    print(f"Testing {title} conversion...")
    output_data = conversion_fn(input_data)
    inv_input_data = inverse_fn(output_data)
    n = input_data.shape[0]
    errors = torch.sum((inv_input_data - input_data)**2, dim=1)
    mse_error = torch.mean(errors)
    max_print = 32 if title.endswith("SPEC4") else 3
    for i in range(min(n, max_print)):
        error = errors[i]
        print(f"  error={error:.4f} from {input_data[i]} to {output_data[i]} back to {inv_input_data[i]}")
    print(f"{title} mean squared error: {mse_error:.12f}")

def test_XYZ_to_xyY():
    input_data = torch.rand(100, 3)
    test_round_trip("XYZ to xyY", input_data, batched_XYZ_to_xyY, batched_xyY_to_XYZ)

def test_sRGB_RGB():
    input_data = torch.rand(100, 3)
    test_round_trip("sRGB to RGB", input_data, batched_sRGB_to_RGB, batched_RGB_to_sRGB)

def test_RGB_XYZ():
    input_data = torch.rand(100, 3)
    test_round_trip("RGB to XYZ", input_data, batched_RGB_to_XYZ, batched_XYZ_to_RGB)

def test_sRGB_XYZ():
    input_data = torch.rand(100, 3)
    test_round_trip("sRGB to XYZ", input_data, batched_sRGB_to_XYZ, batched_XYZ_to_sRGB)

def test_sRGB_SPEC4():
    input_data = torch.rand(100, 3)
    test_round_trip("sRGB to SPEC4", input_data, batched_sRGB_to_SPEC4, batched_SPEC4_to_sRGB)

def test_batched_spectrum_to_XYZ():
    wavelengths = torch.tensor([
        550.0, 650.0,
    ])
    print(f"wavelengths: {wavelengths}")
    spectral_radiance = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
    ])
    print(f"spectral_radiance: {spectral_radiance}")
    xyz = batched_spectrum_to_XYZ(spectral_radiance, wavelengths)
    print(f"xyz.shape: {xyz.shape}")
    print(f"xyz: {xyz}")
    mapped_xyz = tone_map_XYZ(xyz)
    print(f"mapped_xyz: {mapped_xyz}")
    srgb = batched_XYZ_to_sRGB(mapped_xyz)
    print(f"srgb: {srgb}")

def test_all():
    test_luminance()
    test_XYZ_to_xyY()
    test_sRGB_SPEC4()
    test_sRGB_RGB()
    test_RGB_XYZ()
    test_sRGB_XYZ()
    test_batched_spectrum_to_XYZ()

if __name__ == "__main__":
    test_all()
