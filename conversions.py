"""Collection of functions to convert between different color spaces."""

import torch
from torch import Tensor, tensor

import xyz
from xyz import XYZ_to_RGB_right_matrix
import spec3


SPEC3_to_RGB_right_matrix = torch.matmul(spec3.SPEC3_to_XYZ_right_matrix, XYZ_to_RGB_right_matrix)
RGB_to_SPEC3_right_matrix = torch.matmul(xyz.RGB_to_XYZ_right_matrix, spec3.XYZ_to_SPEC3_right_matrix)


#
# Primary color space conversion functions
#

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
    return torch.matmul(rgb, xyz.RGB_to_XYZ_right_matrix)

def batched_XYZ_to_RGB(xyz: Tensor) -> Tensor:
    """Converts CIE XYZ to linear RGB.

    Args:
        xyz: A tensor of CIE XYZ values shaped as (batch_size, 3).

    Returns:
        A tensor of linear RGB values shaped as (batch_size, 3).
    """
    return torch.matmul(xyz, XYZ_to_RGB_right_matrix)

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

def batched_RGB_to_SPEC3(rgb: Tensor, clip: bool = True) -> Tensor:
    """Converts linear RGB to SPEC3.

    Args:
        rgb: A tensor of linear RGB values shaped as (batch_size, 3).

    Returns:
        A tensor of SPEC3 values shaped as (batch_size, 4).
    """
    spec3 = torch.matmul(rgb, RGB_to_SPEC3_right_matrix)
    if clip:
        spec3 = torch.nn.functional.relu(spec3)
    return spec3

def batched_SPEC3_to_XYZ(s3: Tensor) -> Tensor:
    """Converts SPEC3 to CIE XYZ.

    Args:
        s3: A tensor of SPEC3 values shaped as (batch_size, 3).

    Returns:
        A tensor of CIE XYZ values shaped as (batch_size, 3).
    """
    return torch.matmul(s3, spec3.SPEC3_to_XYZ_right_matrix)

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

def batched_sRGB_to_SPEC3(srgb):
    """Converts sRGB to SPEC3.

    Args:
        srgb: A tensor of sRGB values in the range [0, 1]
              shaped as (batch_size, 3).

    Returns:
        A tensor of SPEC3 values shaped as (batch_size, 3).
    """
    return batched_RGB_to_SPEC3(batched_sRGB_to_RGB(srgb))

def batched_SPEC3_to_RGB(s3: Tensor) -> Tensor:
    """Converts SPEC3 to linear RGB.

    Args:
        s3: A tensor of SPEC3 values shaped as (batch_size, 3).

    Returns:
        A tensor of linear RGB values shaped as (batch_size, 3).
    """
    return torch.matmul(s3, SPEC3_to_RGB_right_matrix)

def batched_SPEC3_to_sRGB(s3):
    """Converts SPEC3 to sRGB in the range [0, 1].

    Args:
        s3: A tensor of SPEC3 values shaped as (batch_size, 3).

    Returns:
        A tensor of sRGB values in the range [0, 1] shaped as (batch_size, 3).
    """
    return batched_RGB_to_sRGB(batched_SPEC3_to_RGB(s3))

def batched_XYZ_to_SPEC3(xyz, clip: bool = True):
    """Converts CIE XYZ to SPEC3.

    Args:
        xyz: A tensor of CIE XYZ values shaped as (batch_size, 3).
        clip: Whether to clip the output to non-negative values.

    Returns:
        A tensor of SPEC3 values shaped as (batch_size, 4).
    """
    s3 = torch.matmul(xyz, spec3.XYZ_to_SPEC3_right_matrix)
    if clip:
        s3 = torch.nn.functional.relu(s3)
    return s3

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
    spec3 = batched_XYZ_to_SPEC3(xyz)
    spec32 = batched_XYZ_to_SPEC3(xyz2)
    print(f"spec3: {spec3}")
    print(f"spec32: {spec32}")
    dwavelength = 60.0
    spec3_luminance = torch.sum(spec3, dim=1) * dwavelength
    spec3_luminance2 = torch.sum(spec32, dim=1) * dwavelength
    print(f"spec3_luminance: {spec3_luminance}")
    print(f"spec3_luminance2: {spec3_luminance2}")

def test_round_trip(title, input_data, conversion_fn, inverse_fn):
    """Tests that the conversion and inverse functions are consistent."""
    print(f"Testing {title} conversion...")
    output_data = conversion_fn(input_data)
    inv_input_data = inverse_fn(output_data)
    n = input_data.shape[0]
    errors = torch.sum((inv_input_data - input_data)**2, dim=1)
    mse_error = torch.mean(errors)
    max_print = 32 if title.endswith("SPEC3") else 3
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

def test_sRGB_SPEC3():
    input_data = torch.rand(100, 3)
    test_round_trip("sRGB to SPEC3", input_data, batched_sRGB_to_SPEC3, batched_SPEC3_to_sRGB)

def test_XYZ_SPEC3():
    srgb = torch.rand(100, 3)
    input_data = batched_sRGB_to_XYZ(srgb)
    test_round_trip("XYZ to SPEC3", input_data, batched_XYZ_to_SPEC3, batched_SPEC3_to_XYZ)

def test_all():
    test_luminance()
    test_XYZ_to_xyY()
    test_sRGB_RGB()
    test_RGB_XYZ()
    test_sRGB_XYZ()
    test_sRGB_SPEC3()
    test_XYZ_SPEC3()

if __name__ == "__main__":
    test_all()
