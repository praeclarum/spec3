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

def write_color_table():
    # For each example color we show:
    # 1. The HTML color with the sRGB values
    # 2. The SPEC3 remapped color with SPEC3 values
    # 3. The SPEC3 spectrum (from pyplot)
    srgbs = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5],
        [0.0, 0.0, 0.0],
    ])
    spec3s = conversions.batched_sRGB_to_SPEC3(srgbs)
    roundtrip_srgbs = conversions.batched_SPEC3_to_sRGB(spec3s)
    out_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    def srgb_to_str(srgb: torch.Tensor) -> str:
        isrgb = (srgb * 255).round().int()
        return f"#{isrgb[0]:02x}{isrgb[1]:02x}{isrgb[2]:02x}"
    with open(os.path.join(out_dir, "color_table.md"), "w") as out:
        def write_color(srgb):
            srgb_str = srgb_to_str(srgb)
            out.write(f"<span style='background-color: {srgb_str}; width: 64px; height: 24px; display: inline-block'></span>")
        out.write("# Color Table\n")
        out.write("| Color | sRGB | SPEC3 | Spectrum |\n")
        out.write("| --- | --- | --- | --- |\n")
        for i in range(srgbs.shape[0]):
            srgb = srgbs[i, :]
            srgb_str = srgb_to_str(srgb)
            srgb_id = srgb_str[1:]
            roundtrip_srgb = roundtrip_srgbs[i, :]
            roundtrip_srgb_str = srgb_to_str(roundtrip_srgb)
            spec3 = spec3s[i, :]
            spec3_str = f"({spec3[0]:.3f}, {spec3[1]:.3f}, {spec3[2]:.3f})"
            out.write(f"| ")
            write_color(srgb)
            out.write(f" | {srgb_str} |  {spec3_str} | ")
            spectrum = torch.nn.functional.pad(spec3, (1, 1), value=0.0)
            plot_x = conversions.SPEC3_standard_wavelengths.numpy()
            plt.figure(figsize=(4, 2), facecolor="black")
            # Background to black, text to white
            plt.gca().set_facecolor("black")
            plt.gca().tick_params(axis='x', colors='white')
            plt.gca().tick_params(axis='y', colors='black')
            plt.gca().yaxis.label.set_color('black')
            plt.gca().xaxis.label.set_color('white')
            plt.gca().title.set_color('white')
            # Set border color
            plt.gca().spines['bottom'].set_color('white')
            plt.fill_between(
                x=plot_x,
                y1=0.0, 
                y2=spectrum.numpy(), 
                where= None,
                interpolate= True,
                color= roundtrip_srgb_str,
                alpha= 0.5)
            plt.plot(plot_x, spectrum.numpy(), color=roundtrip_srgb_str)
            plt.ylim(0.0, 1.35)
            spectrum_name = f"spectrum_{srgb_id}.png"
            plt.savefig(os.path.join(images_dir, spectrum_name))
            plt.close()
            out.write(f"<img src='./images/{spectrum_name}' width='200px'> |\n")

if __name__ == "__main__":
    print_matrices()
    write_color_table()
