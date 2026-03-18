#!/usr/bin/env python3
"""
Simple example demonstrating Gaussian absorption line modeling with stellar surface features.

This script shows how different brightness maps affect a single Gaussian absorption line
using the velocity field from velocity_brightness_field.py and spherical harmonics.
"""

import numpy as np
import matplotlib.pyplot as plt
from gaussian_line_modeling import GaussianLineModeler


def main():
    """Simple demonstration of Gaussian line modeling."""
    print("Simple Gaussian Line Modeling Example")
    print("=" * 40)
    
    # Create modeler with vsini = 20 km/s
    modeler = GaussianLineModeler(vsini=5.0, resolution=120)
    
    # Create different brightness maps
    print("Creating brightness maps...")
    
    # 1. Uniform brightness (baseline)
    uniform = modeler.create_uniform_brightness()
    
    # 2. Limb darkening (realistic stellar surface)
    limb_dark = modeler.create_limb_darkening_brightness(u=0.6)
    
    # 3. Spherical harmonic feature (Y_2^0 - zonal pattern)
    harmonic = modeler.create_spherical_harmonic_brightness(l=2, m=0, amplitude=0.4)
    
    # 4. Stellar spot (cool region)
    spot_one = modeler.create_spot_brightness(
        spot_lat=50.0, spot_lon=-22.0, spot_radius=20.0, spot_temperature=0.4
    )
    
    spot_two = modeler.create_spot_brightness(
        spot_lat=40.0, spot_lon=80.0, spot_radius=20.0, spot_temperature=0.4
    )
    
    # Create the comparison plot
    brightness_maps = [
        (uniform, "Uniform"),
        (limb_dark, "Limb Darkening"),
        # (harmonic, "Y₂⁰ Harmonic"),
        (spot_one, "Stellar Spot One"),
        (spot_two, "Stellar Spot Two")
    ]
    
    print("Generating plot...")
    fig = modeler.plot_simple_comparison(brightness_maps, figsize=(18, 12),
                                         vmin=0.0, vmax=1.4)
    save_pdf = True
    save_png = True
    fig_name_pdf = 'simple_gaussian_example.pdf'
    if save_png:
        fig_name_png = fig_name_pdf.replace('.pdf', '.png')
        plt.savefig(fig_name_png, dpi=300, bbox_inches='tight')
        print(f"✓ Example complete! Check '{fig_name_png}'")

    if save_pdf:
        plt.savefig(fig_name_pdf, dpi=300, bbox_inches='tight')
        print(f"✓ Example complete! Check '{fig_name_pdf}'")
    plt.show()
    
    
    # Show quantitative differences
    print("\nLine Profile Analysis:")
    print("-" * 25)
    for brightness, title in brightness_maps:
        v, profile = modeler.compute_gaussian_line_profile(brightness)
        min_int = np.min(profile)
        center_idx = len(profile) // 2
        center_int = profile[center_idx]
        print(f"{title:15}: Min = {min_int:.3f}, Center = {center_int:.3f}")


if __name__ == "__main__":
    main()
