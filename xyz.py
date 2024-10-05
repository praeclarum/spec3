"""CIE XYZ definitions and manipulation functions."""

import torch
from torch import Tensor, tensor


RGB_to_XYZ_left_matrix = torch.tensor([
    [0.49000, 0.31000, 0.20000],
    [0.17697, 0.81240, 0.01063],
    [0.00000, 0.01000, 0.99000],
])
RGB_to_XYZ_right_matrix = RGB_to_XYZ_left_matrix.T

XYZ_to_RGB_right_matrix = torch.inverse(RGB_to_XYZ_right_matrix)


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

if __name__ == "__main__":
    test_batched_spectrum_to_XYZ()
