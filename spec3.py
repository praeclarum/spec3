"""SPEC3 definitions and manipulation functions."""

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

