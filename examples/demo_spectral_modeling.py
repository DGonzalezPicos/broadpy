#!/usr/bin/env python3
"""
Quick demonstration of spectral line modeling with stellar surface features.

This script shows the key capabilities of the spectral line modeling system
for cool star analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from spherical_harmonics import SphericalHarmonicsPlotter
from spectral_line_modeling import SpectralLineModeler


def demo_intensity_maps():
    """Demonstrate intensity map generation."""
    print("1. Generating intensity maps for stellar disc...")
    
    plotter = SphericalHarmonicsPlotter(resolution=80)
    
    # Plot a few key harmonics as intensity maps
    harmonics_to_plot = [(2, 0), (3, 1), (4, 2)]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw=dict(projection='polar'))
    
    for i, (l, m) in enumerate(harmonics_to_plot):
        r, theta, intensity = plotter.get_intensity_map(l, m)
        
        im = axes[i].pcolormesh(theta, r, intensity, cmap='inferno', shading='auto')
        axes[i].set_title(f'Y_{l}^{m}', fontsize=12, fontweight='bold', pad=20)
        axes[i].set_ylim(0, 1)
        plt.colorbar(im, ax=axes[i], shrink=0.8, label='Intensity')
    
    plt.suptitle('Stellar Disc Intensity Maps from Spherical Harmonics', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('intensity_maps_demo.png', dpi=300, bbox_inches='tight')
    plt.show()


def demo_spectral_effects():
    """Demonstrate spectral line effects from surface features."""
    print("\n2. Modeling spectral line effects...")
    
    modeler = SpectralLineModeler(v_rot=40.0, v_micro=1.5, v_macro=3.0)
    
    # Compare different surface features
    features = [
        (2, 0, 0.3, "Zonal pattern (Y₂⁰)"),
        (3, 1, 0.25, "Tesseral pattern (Y₃¹)"),
        (4, 2, 0.2, "Complex pattern (Y₄²)")
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for i, (l, m, amplitude, title) in enumerate(features):
        # Surface map
        theta, phi, intensity = modeler.create_surface_intensity_map(l, m, feature_amplitude=amplitude)
        r = np.sin(theta)
        
        # Plot surface
        im = axes[0, i].pcolormesh(phi, r, intensity, cmap='inferno', shading='auto')
        axes[0, i].set_title(f'{title}\nSurface Map', fontsize=11, fontweight='bold')
        axes[0, i].set_aspect('equal')
        axes[0, i].set_xlabel('φ')
        axes[0, i].set_ylabel('r')
        
        # Plot line profile
        v, line_profile = modeler.compute_line_profile(theta, phi, intensity)
        axes[1, i].plot(v, line_profile, 'b-', linewidth=2, label='With features')
        
        # Add uniform surface for comparison
        intensity_uniform = np.ones_like(intensity) * modeler._compute_limb_darkening(theta)
        v_uniform, line_uniform = modeler.compute_line_profile(theta, phi, intensity_uniform)
        axes[1, i].plot(v_uniform, line_uniform, 'r--', linewidth=2, label='Uniform')
        
        axes[1, i].set_xlabel('Velocity (km/s)')
        axes[1, i].set_ylabel('Normalized Intensity')
        axes[1, i].set_title('Line Profile', fontsize=11, fontweight='bold')
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].legend()
        axes[1, i].set_ylim(0, 1.1)
    
    plt.suptitle('Surface Features and Spectral Line Effects', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('spectral_effects_demo.png', dpi=300, bbox_inches='tight')
    plt.show()


def demo_spot_evolution():
    """Demonstrate spot rotation effects."""
    print("\n3. Modeling spot rotation effects...")
    
    modeler = SpectralLineModeler(v_rot=60.0, v_micro=2.0, v_macro=4.0)
    
    # Different spot longitudes
    spot_longitudes = [0, 45, 90, 135, 180]
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    for i, lon in enumerate(spot_longitudes):
        # Create spot
        theta, phi, intensity = modeler.create_spot_model(
            spot_lat=30.0, spot_lon=lon, spot_radius=20.0, spot_temperature=0.6
        )
        
        # Surface map
        r = np.sin(theta)
        im = axes[0, i].pcolormesh(phi, r, intensity, cmap='inferno', shading='auto')
        axes[0, i].set_title(f'Spot at {lon}°', fontsize=10, fontweight='bold')
        axes[0, i].set_aspect('equal')
        
        # Line profile
        v, line_profile = modeler.compute_line_profile(theta, phi, intensity)
        axes[1, i].plot(v, line_profile, 'b-', linewidth=2)
        axes[1, i].set_xlabel('Velocity (km/s)')
        axes[1, i].set_ylabel('Intensity')
        axes[1, i].set_title(f'φ = {lon}°', fontsize=10, fontweight='bold')
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].set_ylim(0, 1.1)
    
    plt.suptitle('Spot Rotation Effects on Spectral Lines', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('spot_rotation_demo.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main demonstration function."""
    print("Spectral Line Modeling Demo for Cool Star Analysis")
    print("=" * 50)
    
    # Run demonstrations
    demo_intensity_maps()
    demo_spectral_effects()
    demo_spot_evolution()
    
    print("\n✓ Demo complete! Check the generated PNG files.")
    print("\nKey features demonstrated:")
    print("• Intensity maps in polar coordinates (r, θ)")
    print("• Spherical harmonic surface features")
    print("• Spectral line profile calculations")
    print("• Doppler broadening effects")
    print("• Spot modeling and rotation effects")


if __name__ == "__main__":
    main()
