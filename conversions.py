"""Collection of functions to convert between different color spaces."""

import torch
from torch import Tensor, tensor


#
# Color Space Conversion Matrices
#

RGB_to_XYZ_left_matrix = torch.tensor([
    [0.49000, 0.31000, 0.20000],
    [0.17697, 0.81240, 0.01063],
    [0.00000, 0.01000, 0.99000],
])
RGB_to_XYZ_right_matrix = RGB_to_XYZ_left_matrix.T

XYZ_to_RGB_left_matrix = torch.inverse(RGB_to_XYZ_left_matrix)
XYZ_to_RGB_right_matrix = torch.inverse(RGB_to_XYZ_right_matrix)

SPEC4_standard_wavelengths = tensor([400.0, 460.0, 520.0, 580.0, 640.0, 700.0])

SPEC4_wavelengths = tensor([351.522644042969, 440.371154785156, 546.317138671875, 630.646789550781,
        798.552429199219, 835.963256835938])
XYZ_to_SPEC4_matrix = tensor([[ 3.399193883524e-05, -5.034552421421e-03,  1.453711930662e-02,
         -9.536559693515e-03],
        [-8.494817302562e-05,  1.257958170027e-02, -5.503239575773e-03,
         -6.991393864155e-03],
        [ 5.900643765926e-03,  9.003557497635e-04, -2.944991458207e-03,
         -3.856008639559e-03]])
SPEC4_to_XYZ_matrix = tensor([[3.501516723633e+01, 1.896471738815e+00, 1.692986297607e+02],
        [3.570357894897e+01, 9.370123291016e+01, 1.143284440041e+00],
        [8.107255554199e+01, 3.244654846191e+01, 7.815709977876e-05],
        [1.218585481411e-07, 8.563614755985e-06, 1.442840641160e-20]])

SPEC4_to_RGB_matrix = torch.matmul(SPEC4_to_XYZ_matrix, XYZ_to_RGB_right_matrix)
RGB_to_SPEC4_matrix = torch.matmul(RGB_to_XYZ_right_matrix, XYZ_to_SPEC4_matrix)


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
    return torch.matmul(rgb, RGB_to_XYZ_right_matrix)

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
    return torch.matmul(spec4, SPEC4_to_XYZ_matrix)

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
    return torch.matmul(spec4, SPEC4_to_RGB_matrix)

def batched_SPEC4_to_sRGB(spec4):
    """Converts SPEC4 to sRGB in the range [0, 1].

    Args:
        spec4: A tensor of SPEC4 values shaped as (batch_size, 4).

    Returns:
        A tensor of sRGB values in the range [0, 1] shaped as (batch_size, 3).
    """
    return batched_RGB_to_sRGB(batched_SPEC4_to_RGB(spec4))

def batched_XYZ_to_SPEC4(xyz, clip: bool = True):
    """Converts CIE XYZ to SPEC4.

    Args:
        xyz: A tensor of CIE XYZ values shaped as (batch_size, 3).
        clip: Whether to clip the output to non-negative values.

    Returns:
        A tensor of SPEC4 values shaped as (batch_size, 4).
    """
    spec4 = torch.matmul(xyz, XYZ_to_SPEC4_matrix)
    if clip:
        spec4 = torch.nn.functional.relu(spec4)
    return spec4

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

def test_XYZ_SPEC4():
    srgb = torch.rand(100, 3)
    input_data = batched_sRGB_to_XYZ(srgb)
    test_round_trip("XYZ to SPEC4", input_data, batched_XYZ_to_SPEC4, batched_SPEC4_to_XYZ)

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
    test_sRGB_RGB()
    test_RGB_XYZ()
    test_sRGB_XYZ()
    test_batched_spectrum_to_XYZ()
    test_sRGB_SPEC4()
    test_XYZ_SPEC4()

def print_matrix(matrix, name, in_names, out_names):
    """Writes the code to execute out_names = matrix * in_names."""
    in_params = ", ".join(in_names)
    print(f"def {name}({in_params}):")
    for j, out_name in enumerate(out_names):
        out_expr = ""
        for i, in_name in enumerate(in_names):
            value = matrix[i, j].item()
            if i == 0:
                if value < 0.0:
                    prefix = "-"
                else:
                    prefix = ""
            else:
                if value < 0.0:
                    prefix = " - "
                else:
                    prefix = " + "
            out_expr += prefix + f"{abs(value):.12f} * {in_name}"
        print(f"    {out_name} = {out_expr}")
    print(f"    return {', '.join(out_names)}")

def print_vectorized_matrix(matrix, name, in_name):
    """Writes the code to execute out_vector = matrix * in_vector."""
    in_channels = matrix.shape[0]
    out_channels = matrix.shape[1]
    print(f"{name}({in_name}: vec{in_channels}<f32>) -> vec{out_channels}<f32> {{")
    print(f"    const {name}: mat{out_channels}x{in_channels}<f32> = mat{out_channels}x{in_channels}(")
    for j in range(out_channels):
        print("        ", end="")
        for i in range(in_channels):
            value = matrix[i, j].item()
            if value < 0.0:
                prefix = "-"
            else:
                prefix = ""
            print(f"{prefix}{abs(value):.12f}, ", end="")
        print()
    print("    );")
    print(f"    return {name} * {in_name};")
    print("}")

def test_mat44():
    XYZ_to_SPEC4_matrix

    xyz = torch.tensor([0.25, 0.5, 0.85])

    XYZ_to_SPEC4_square_matrix = torch.cat([
        XYZ_to_SPEC4_matrix,
        torch.ones(1, XYZ_to_SPEC4_matrix.shape[1])], dim=0)
    xyz4 = torch.cat([xyz, torch.tensor([0.0])])
    spec44 = XYZ_to_SPEC4_square_matrix.T @ xyz4

    SPEC4_to_XYZ_square_matrix = torch.inverse(XYZ_to_SPEC4_square_matrix.T)
    SPEC4_to_XYZ_square_matrix @ spec44

    SPEC4_to_XYZ_matrix
    SPEC4_to_XYZ_square_matrix_p = torch.cat([
        SPEC4_to_XYZ_matrix,
        torch.ones(SPEC4_to_XYZ_matrix.shape[0], 1)], dim=1)

    XYZ_to_SPEC4_matrix
    XYZ_to_SPEC4_square_matrix_p = torch.inverse(SPEC4_to_XYZ_square_matrix_p)
    xyz41 = torch.cat([xyz, torch.tensor([0.0])])
    xyz41 @ XYZ_to_SPEC4_square_matrix_p
    xyz4 @ XYZ_to_SPEC4_square_matrix_p
    xyz @ XYZ_to_SPEC4_matrix

def print_matrices():
    print("```python")
    print_matrix(XYZ_to_SPEC4_matrix, "xyz_to_spec4", ["x", "y", "z"], ["sx", "sy", "sz", "sw"])
    print_matrix(SPEC4_to_XYZ_matrix, "spec4_to_xyz", ["sx", "sy", "sz", "sw"], ["x", "y", "z"])
    print()
    print_matrix(RGB_to_SPEC4_matrix, "rgb_to_spec4", ["r", "g", "b"], ["sx", "sy", "sz", "sw"])
    print_matrix(SPEC4_to_RGB_matrix, "spec4_to_rgb", ["sx", "sy", "sz", "sw"], ["r", "g", "b"])
    print("```")
    print()
    print("```javascript")
    print_vectorized_matrix(XYZ_to_SPEC4_matrix, "xyz_to_spec4", "xyz")
    print_vectorized_matrix(SPEC4_to_XYZ_matrix, "spec4_to_xyz", "spec4")
    print()
    print_vectorized_matrix(RGB_to_SPEC4_matrix, "rgb_to_spec4", "rgb")
    print_vectorized_matrix(SPEC4_to_RGB_matrix, "spec4_to_rgb", "spec4")
    print("```")

if __name__ == "__main__":
    test_all()
    # print_matrices()
