"""Generates documentation for SPEC3"""
import os
import sys
import re
from matplotlib import pyplot as plt
import torch

import conversions

def print_matrix(right_matrix, name, in_names, out_names):
    """Writes the code to execute out_names = matrix * in_names."""
    in_params = ", ".join(in_names)
    print(f"def {name}({in_params}):")
    for j, out_name in enumerate(out_names):
        out_expr = ""
        for i, in_name in enumerate(in_names):
            value = right_matrix[i, j].item()
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

def print_vectorized_matrix(right_matrix, name, in_name):
    """Writes the code to execute out_vector = matrix * in_vector."""
    in_channels = right_matrix.shape[0]
    out_channels = right_matrix.shape[1]
    print(f"fn {name}({in_name}: vec{in_channels}<f32>) -> vec{out_channels}<f32> {{")
    print(f"    const {name}: mat{out_channels}x{in_channels}<f32> = mat{out_channels}x{in_channels}(")
    for j in range(out_channels):
        print("        ", end="")
        for i in range(in_channels):
            value = right_matrix[i, j].item()
            if value < 0.0:
                prefix = "-"
            else:
                prefix = ""
            print(f"{prefix}{abs(value):.12f}, ", end="")
        print()
    print("    );")
    print(f"    return {name} * {in_name};")
    print("}")

def print_matrices():
    print("```python")
    print_matrix(conversions.XYZ_to_SPEC3_right_matrix, "xyz_to_spec3", ["x", "y", "z"], ["sx", "sy", "sz"])
    print_matrix(conversions.SPEC3_to_XYZ_right_matrix, "spec3_to_xyz", ["sx", "sy", "sz"], ["x", "y", "z"])
    print()
    print_matrix(conversions.RGB_to_SPEC3_right_matrix, "rgb_to_spec3", ["r", "g", "b"], ["sx", "sy", "sz"])
    print_matrix(conversions.SPEC3_to_RGB_right_matrix, "spec3_to_rgb", ["sx", "sy", "sz"], ["r", "g", "b"])
    print("```")
    print()
    print("```rust")
    print_vectorized_matrix(conversions.XYZ_to_SPEC3_right_matrix, "xyz_to_spec3", "xyz")
    print_vectorized_matrix(conversions.SPEC3_to_XYZ_right_matrix, "spec3_to_xyz", "spec3")
    print()
    print_vectorized_matrix(conversions.RGB_to_SPEC3_right_matrix, "rgb_to_spec3", "rgb")
    print_vectorized_matrix(conversions.SPEC3_to_RGB_right_matrix, "spec3_to_rgb", "spec3")
    print("```")

if __name__ == "__main__":
    print_matrices()
