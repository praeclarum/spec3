"""SPEC3 definitions and manipulation functions."""

import math
import torch
from torch import Tensor, tensor

import xyz


SPEC3_standard_wavelengths = tensor([348.0, 438.0, 542.0, 644.0, 760.0])

SPEC3_scale = 100.0

def get_optimal_SPEC3_to_XYZ_right_matrix(wavelengths: Tensor) -> Tensor:
    # Matching, m, is shaped (num_wavelengths, 3)
    # dWavelength, dw, is shaped (num_wavelengths-1,)
    # a = m[1]
    # b = m[2]
    # c = m[3]
    # integral = (0 + a)/2*dw[0] + (a + b)/2*dw[1] + (b + c)/2*dw[2] + (c + 0)/2*dw[3]
    # integral = a*(dw[0] + dw[1])/2 + b*(dw[1] + dw[2])/2 + c*(dw[2] + dw[3])/2
    # mean_dw[i] = (dw[i] + dw[i+1])/2
    m = xyz.xyz_color_matching(wavelengths)
    dw = wavelengths[1:] - wavelengths[:-1]
    mean_dw = (dw[:-1] + dw[1:])/2
    a = m[1, :]
    b = m[2, :]
    c = m[3, :]
    matrix = torch.stack([
        a*mean_dw[0],
        b*mean_dw[1],
        c*mean_dw[2],
    ], dim=1).T
    return matrix

SPEC3_to_XYZ_right_matrix = get_optimal_SPEC3_to_XYZ_right_matrix(SPEC3_standard_wavelengths) / SPEC3_scale
XYZ_to_SPEC3_right_matrix = torch.inverse(SPEC3_to_XYZ_right_matrix)

def mix(a, b, u):
    return a * (1.0 - u) + b * u

def get_radiance(sx, sy, sz, wavelength):
    if wavelength <= 348.0:
        return 0.0
    if wavelength <= 438.0:
        u = (wavelength - 348.0) / (438.0 - 348.0)
        return mix(0.0, sx, u)
    if wavelength <= 542.0:
        u = (wavelength - 438.0) / (542.0 - 438.0)
        return mix(sx, sy, u)
    if wavelength <= 644.0:
        u = (wavelength - 542.0) / (644.0 - 542.0)
        return mix(sy, sz, u)
    if wavelength <= 760.0:
        u = (wavelength - 644.0) / (760.0 - 644.0)
        return mix(sz, 0.0, u)
    return 0.0

def get_radiance_with_wavelengths(s3: Tensor, wavelengths: Tensor, wavelength: float):
    wa = wavelengths[0]
    wx = wavelengths[1]
    wy = wavelengths[2]
    wz = wavelengths[3]
    wo = wavelengths[4]
    sx, sy, sz = s3
    if wavelength <= wa:
        return 0.0
    if wavelength <= wx:
        u = (wavelength - wa) / (wx - wa)
        return mix(0.0, sx, u)
    if wavelength <= wy:
        u = (wavelength - wx) / (wy - wx)
        return mix(sx, sy, u)
    if wavelength <= wz:
        u = (wavelength - wy) / (wz - wy)
        return mix(sy, sz, u)
    if wavelength <= wo:
        u = (wavelength - wz) / (wo - wz)
        return mix(sz, 0.0, u)
    return 0.0

def doppler_shift(s3: Tensor, beta: float):
    """Doppler shift the SPEC3 values by beta.
    Beta is the ratio relative velocity/speed of light.
    When beta > 0, the source is moving away from the receiver.
    When beta < 0, the source is moving towards the receiver.

    Args:
        s3: The SPEC3 value of the source.
        beta: The ratio relative velocity/speed of light.

    Returns:
        The SPEC3 values of the source color 
        after the Doppler shift has been applied.
    """
    # wavelength_r/wavelength_s = sqrt((1 + beta)/(1 - beta))
    wavelength_scale = math.sqrt((1 + beta) / (1 - beta))
    received_wavelengths = SPEC3_standard_wavelengths * wavelength_scale
    new_sx = get_radiance_with_wavelengths(s3, received_wavelengths, SPEC3_standard_wavelengths[1])
    new_sy = get_radiance_with_wavelengths(s3, received_wavelengths, SPEC3_standard_wavelengths[2])
    new_sz = get_radiance_with_wavelengths(s3, received_wavelengths, SPEC3_standard_wavelengths[3])
    return tensor([new_sx, new_sy, new_sz])

def test_doppler_shift():
    print("-"*40)
    print("Doppler shift test")
    print("-"*40)
    s3 = tensor([0.0, 0.795, 0.136])
    print(f"Incoming s3: {s3}")
    betas = [-0.7, -0.5, -0.1, 0.0, 0.1, 0.36, 0.5]
    for beta in betas:
        recv_s3 = doppler_shift(s3, beta)
        print(f"  beta: {beta:.2f}, Received: {recv_s3}")

if __name__ == "__main__":
    test_doppler_shift()
