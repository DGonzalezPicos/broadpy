#!/usr/bin/env python3
"""
Demonstration script for the SphericalHarmonicsPlotter class.

This script shows how to use the SphericalHarmonicsPlotter class to generate
and visualize spherical harmonics with different options.
"""

import matplotlib.pyplot as plt
from spherical_harmonics import SphericalHarmonicsPlotter


def demo_basic_plotting():
    """Demonstrate basic spherical harmonics plotting."""
    print("Creating SphericalHarmonicsPlotter instance...")
    plotter = SphericalHarmonicsPlotter(resolution=60)
    
    print("Generating and plotting first 20 spherical harmonics...")
    fig = plotter.plot_harmonics(figsize=(16, 12), colormap='RdBu_r')
    plt.savefig('spherical_harmonics_grid.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return plotter


def demo_single_harmonic(plotter: SphericalHarmonicsPlotter):
    """Demonstrate single harmonic plotting."""
    print("\nPlotting individual harmonics...")
    
    # Plot Y_2^0 (zonal harmonic)
    fig1 = plotter.plot_single_harmonic(l=2, m=0, colormap='viridis')
    plt.savefig('spherical_harmonic_Y20.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot Y_3^1 (tesseral harmonic)
    fig2 = plotter.plot_single_harmonic(l=3, m=1, colormap='plasma')
    plt.savefig('spherical_harmonic_Y31.png', dpi=300, bbox_inches='tight')
    plt.show()


def demo_2d_projections(plotter: SphericalHarmonicsPlotter):
    """Demonstrate 2D projection plotting."""
    print("\nPlotting 2D projections...")
    
    # Plot 2D projections of Y_1^1
    fig = plotter.plot_2d_projection(l=1, m=1, colormap='coolwarm')
    plt.savefig('spherical_harmonic_Y11_2d.png', dpi=300, bbox_inches='tight')
    plt.show()


def demo_imaginary_parts(plotter: SphericalHarmonicsPlotter):
    """Demonstrate plotting imaginary parts."""
    print("\nPlotting imaginary parts...")
    
    # Plot imaginary part of Y_2^2
    fig = plotter.plot_single_harmonic(l=2, m=2, show_phase=False, colormap='seismic')
    plt.savefig('spherical_harmonic_Y22_imag.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main demonstration function."""
    print("Spherical Harmonics Visualization Demo")
    print("=" * 40)
    
    # Basic plotting
    plotter = demo_basic_plotting()
    
    # Single harmonic examples
    demo_single_harmonic(plotter)
    
    # 2D projections
    demo_2d_projections(plotter)
    
    # Imaginary parts
    demo_imaginary_parts(plotter)
    
    print("\nDemo complete! Check the generated PNG files for the visualizations.")


if __name__ == "__main__":
    main()
