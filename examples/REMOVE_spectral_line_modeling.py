"""
Spectral Line Modeling with Stellar Surface Features

This script demonstrates how different stellar surface features (spots, 
bright regions) affect the shape of spectral absorption lines through
Doppler broadening and intensity variations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from scipy.special import voigt_profile
from typing import Tuple, List, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Import our spherical harmonics class
from spherical_harmonics import SphericalHarmonicsPlotter


class SpectralLineModeler:
    """
    A class for modeling spectral absorption lines with stellar surface features.
    
    This class combines spherical harmonic surface maps with Doppler broadening
    to simulate how stellar spots and features affect spectral line profiles.
    """
    
    def __init__(self, 
                 v_rot: float = 50.0,
                 v_micro: float = 2.0,
                 v_macro: float = 5.0,
                 resolution: int = 100) -> None:
        """
        Initialize the spectral line modeler.
        
        Parameters
        ----------
        v_rot : float, optional
            Stellar rotation velocity in km/s, by default 50.0
        v_micro : float, optional
            Microturbulent velocity in km/s, by default 2.0
        v_macro : float, optional
            Macroturbulent velocity in km/s, by default 5.0
        resolution : int, optional
            Angular resolution for surface grid, by default 100
        """
        self.v_rot = v_rot
        self.v_micro = v_micro
        self.v_macro = v_macro
        self.resolution = resolution
        
        # Create surface coordinate grids
        self.theta_surf, self.phi_surf = self._create_surface_grid()
        
        # Initialize spherical harmonics plotter
        self.harmonics_plotter = SphericalHarmonicsPlotter(resolution=resolution)
    
    def _create_surface_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create surface coordinate grids for stellar surface.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            theta and phi coordinate arrays
        """
        theta = np.linspace(0, np.pi, self.resolution)
        phi = np.linspace(0, 2 * np.pi, self.resolution)
        return theta, phi
    
    def _compute_doppler_velocity(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        Compute Doppler velocity for each surface element.
        
        Parameters
        ----------
        theta : np.ndarray
            Polar angle array
        phi : np.ndarray
            Azimuthal angle array
            
        Returns
        -------
        np.ndarray
            Doppler velocity array in km/s
        """
        # Projected rotation velocity: v_rot * sin(theta) * cos(phi)
        v_proj = self.v_rot * np.sin(theta) * np.cos(phi)
        return v_proj
    
    def _compute_limb_darkening(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute limb darkening factor for each surface element.
        
        Parameters
        ----------
        theta : np.ndarray
            Polar angle array
            
        Returns
        -------
        np.ndarray
            Limb darkening factor array
        """
        # Simple linear limb darkening: I = I0 * (1 - u + u*cos(theta))
        u = 0.6  # Limb darkening coefficient
        mu = np.cos(theta)
        return 1 - u + u * mu
    
    def create_surface_intensity_map(self, 
                                   l: int, 
                                   m: int,
                                   base_intensity: float = 1.0,
                                   feature_amplitude: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create surface intensity map using spherical harmonics.
        
        Parameters
        ----------
        l : int
            Degree of spherical harmonic
        m : int
            Order of spherical harmonic
        base_intensity : float, optional
            Base intensity level, by default 1.0
        feature_amplitude : float, optional
            Amplitude of surface features, by default 0.3
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            theta, phi, and intensity arrays
        """
        theta_grid, phi_grid = np.meshgrid(self.theta_surf, self.phi_surf, indexing='ij')
        
        # Compute spherical harmonic
        harmonic = self.harmonics_plotter._compute_spherical_harmonic(l, m)
        harmonic_real = np.real(harmonic)
        
        # Normalize harmonic to [-1, 1] range
        harmonic_norm = harmonic_real / np.max(np.abs(harmonic_real))
        
        # Create intensity map with features
        intensity = base_intensity + feature_amplitude * harmonic_norm
        
        # Apply limb darkening
        limb_darkening = self._compute_limb_darkening(theta_grid)
        intensity *= limb_darkening
        
        return theta_grid, phi_grid, intensity
    
    def compute_line_profile(self, 
                           theta: np.ndarray, 
                           phi: np.ndarray, 
                           intensity: np.ndarray,
                           v_center: float = 0.0,
                           gamma: float = 0.1,
                           sigma: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute spectral line profile considering surface features.
        
        Parameters
        ----------
        theta : np.ndarray
            Polar angle array
        phi : np.ndarray
            Azimuthal angle array
        intensity : np.ndarray
            Surface intensity array
        v_center : float, optional
            Central velocity in km/s, by default 0.0
        gamma : float, optional
            Lorentzian width parameter, by default 0.1
        sigma : float, optional
            Gaussian width parameter, by default 0.05
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Velocity array and line profile
        """
        # Create velocity grid
        v_max = self.v_rot + 3 * self.v_macro
        v = np.linspace(-v_max, v_max, 200)
        
        # Compute Doppler velocities for each surface element
        v_doppler = self._compute_doppler_velocity(theta, phi)
        
        # Initialize line profile
        line_profile = np.zeros_like(v)
        
        # Add contribution from each surface element
        for i in range(theta.shape[0]):
            for j in range(theta.shape[1]):
                # Skip elements behind the star (theta > pi/2)
                if theta[i, j] > np.pi/2:
                    continue
                
                # Compute element area (solid angle)
                dtheta = np.pi / self.resolution
                dphi = 2 * np.pi / self.resolution
                domega = np.sin(theta[i, j]) * dtheta * dphi
                
                # Compute element contribution
                v_element = v_center + v_doppler[i, j]
                element_intensity = intensity[i, j] * domega
                
                # Add Voigt profile contribution
                for k, v_val in enumerate(v):
                    voigt = voigt_profile(v_val - v_element, sigma, gamma)
                    line_profile[k] += element_intensity * voigt
        
        # Normalize profile
        if np.max(line_profile) > 0:
            line_profile = line_profile / np.max(line_profile)
        
        return v, line_profile
    
    def plot_surface_and_line(self, 
                             l: int, 
                             m: int,
                             feature_amplitude: float = 0.3,
                             figsize: Tuple[int, int] = (16, 8)) -> plt.Figure:
        """
        Plot surface intensity map and corresponding line profile.
        
        Parameters
        ----------
        l : int
            Degree of spherical harmonic
        m : int
            Order of spherical harmonic
        feature_amplitude : float, optional
            Amplitude of surface features, by default 0.3
        figsize : Tuple[int, int], optional
            Figure size in inches, by default (16, 8)
            
        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        # Create surface intensity map
        theta, phi, intensity = self.create_surface_intensity_map(l, m, feature_amplitude=feature_amplitude)
        
        # Compute line profile
        v, line_profile = self.compute_line_profile(theta, phi, intensity)
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Surface intensity map (polar view)
        r = np.sin(theta)  # Project to stellar disc
        
        # Create proper coordinate arrays for pcolormesh
        phi_edges = np.linspace(0, 2*np.pi, phi.shape[1]+1)
        r_edges = np.linspace(0, 1, r.shape[0]+1)
        phi_centers = (phi_edges[:-1] + phi_edges[1:]) / 2
        r_centers = (r_edges[:-1] + r_edges[1:]) / 2
        
        im1 = ax1.pcolormesh(phi_edges, r_edges, intensity, cmap='inferno', shading='flat')
        ax1.set_title(f'Surface Intensity Map Y_{l}^{m}', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Azimuthal Angle φ', fontsize=10)
        ax1.set_ylabel('Radial Distance r', fontsize=10)
        ax1.set_aspect('equal')
        plt.colorbar(im1, ax=ax1, shrink=0.8, label='Intensity')
        
        # Plot 2: Surface intensity map (3D view)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z = np.cos(theta)
        
        # Only show visible hemisphere
        visible = theta <= np.pi/2
        x_vis = np.where(visible, x, np.nan)
        y_vis = np.where(visible, y, np.nan)
        z_vis = np.where(visible, z, np.nan)
        intensity_vis = np.where(visible, intensity, np.nan)
        
        im2 = ax2.scatter(x_vis.flatten(), y_vis.flatten(), 
                         c=intensity_vis.flatten(), cmap='inferno', s=1)
        ax2.set_title('3D Surface View', fontsize=12, fontweight='bold')
        ax2.set_xlabel('X', fontsize=10)
        ax2.set_ylabel('Y', fontsize=10)
        ax2.set_aspect('equal')
        ax2.set_xlim(-1.1, 1.1)
        ax2.set_ylim(-1.1, 1.1)
        
        # Add stellar disc outline
        circle = Circle((0, 0), 1, fill=False, color='white', linewidth=2)
        ax2.add_patch(circle)
        
        # Plot 3: Line profile
        ax3.plot(v, line_profile, 'b-', linewidth=2, label=f'Y_{l}^{m} features')
        ax3.set_xlabel('Velocity (km/s)', fontsize=12)
        ax3.set_ylabel('Normalized Intensity', fontsize=12)
        ax3.set_title('Spectral Line Profile', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_ylim(0, 1.1)
        
        # Plot 4: Line profile comparison (with and without features)
        # Compute line profile without features
        intensity_uniform = np.ones_like(intensity) * self._compute_limb_darkening(theta)
        v_uniform, line_uniform = self.compute_line_profile(theta, phi, intensity_uniform)
        
        ax4.plot(v, line_profile, 'b-', linewidth=2, label=f'With Y_{l}^{m} features')
        ax4.plot(v_uniform, line_uniform, 'r--', linewidth=2, label='Uniform surface')
        ax4.set_xlabel('Velocity (km/s)', fontsize=12)
        ax4.set_ylabel('Normalized Intensity', fontsize=12)
        ax4.set_title('Line Profile Comparison', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_ylim(0, 1.1)
        
        plt.suptitle(f'Stellar Surface Features and Spectral Line Modeling\n'
                    f'Y_{l}^{m} with amplitude {feature_amplitude:.1f}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def create_spot_model(self, 
                         spot_lat: float = 30.0,
                         spot_lon: float = 0.0,
                         spot_radius: float = 20.0,
                         spot_temperature: float = 0.7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a simple spot model on the stellar surface.
        
        Parameters
        ----------
        spot_lat : float, optional
            Spot latitude in degrees, by default 30.0
        spot_lon : float, optional
            Spot longitude in degrees, by default 0.0
        spot_radius : float, optional
            Spot radius in degrees, by default 20.0
        spot_temperature : float, optional
            Spot temperature relative to photosphere, by default 0.7
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            theta, phi, and intensity arrays
        """
        theta_grid, phi_grid = np.meshgrid(self.theta_surf, self.phi_surf, indexing='ij')
        
        # Convert to degrees
        theta_deg = np.degrees(theta_grid)
        phi_deg = np.degrees(phi_grid)
        
        # Create spot
        spot_lat_rad = np.radians(spot_lat)
        spot_lon_rad = np.radians(spot_lon)
        spot_radius_rad = np.radians(spot_radius)
        
        # Compute angular distance from spot center
        cos_distance = (np.sin(np.radians(theta_deg)) * np.sin(spot_lat_rad) + 
                       np.cos(np.radians(theta_deg)) * np.cos(spot_lat_rad) * 
                       np.cos(np.radians(phi_deg) - spot_lon_rad))
        
        # Create spot mask
        spot_mask = cos_distance > np.cos(spot_radius_rad)
        
        # Create intensity map
        intensity = np.ones_like(theta_grid)
        intensity[spot_mask] = spot_temperature
        
        # Apply limb darkening
        limb_darkening = self._compute_limb_darkening(theta_grid)
        intensity *= limb_darkening
        
        return theta_grid, phi_grid, intensity
    
    def plot_spot_model(self, 
                       spot_lat: float = 30.0,
                       spot_lon: float = 0.0,
                       spot_radius: float = 20.0,
                       spot_temperature: float = 0.7,
                       figsize: Tuple[int, int] = (16, 8)) -> plt.Figure:
        """
        Plot spot model and corresponding line profile.
        
        Parameters
        ----------
        spot_lat : float, optional
            Spot latitude in degrees, by default 30.0
        spot_lon : float, optional
            Spot longitude in degrees, by default 0.0
        spot_radius : float, optional
            Spot radius in degrees, by default 20.0
        spot_temperature : float, optional
            Spot temperature relative to photosphere, by default 0.7
        figsize : Tuple[int, int], optional
            Figure size in inches, by default (16, 8)
            
        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        # Create spot model
        theta, phi, intensity = self.create_spot_model(spot_lat, spot_lon, spot_radius, spot_temperature)
        
        # Compute line profile
        v, line_profile = self.compute_line_profile(theta, phi, intensity)
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Spot map (polar view)
        r = np.sin(theta)
        
        # Create proper coordinate arrays for pcolormesh
        phi_edges = np.linspace(0, 2*np.pi, phi.shape[1]+1)
        r_edges = np.linspace(0, 1, r.shape[0]+1)
        
        im1 = ax1.pcolormesh(phi_edges, r_edges, intensity, cmap='inferno', shading='flat')
        ax1.set_title(f'Stellar Spot Model\nLat: {spot_lat}°, Lon: {spot_lon}°, R: {spot_radius}°', 
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('Azimuthal Angle φ', fontsize=10)
        ax1.set_ylabel('Radial Distance r', fontsize=10)
        ax1.set_aspect('equal')
        plt.colorbar(im1, ax=ax1, shrink=0.8, label='Relative Intensity')
        
        # Plot 2: Spot map (3D view)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        z = np.cos(theta)
        
        visible = theta <= np.pi/2
        x_vis = np.where(visible, x, np.nan)
        y_vis = np.where(visible, y, np.nan)
        intensity_vis = np.where(visible, intensity, np.nan)
        
        im2 = ax2.scatter(x_vis.flatten(), y_vis.flatten(), 
                         c=intensity_vis.flatten(), cmap='inferno', s=1)
        ax2.set_title('3D Spot View', fontsize=12, fontweight='bold')
        ax2.set_xlabel('X', fontsize=10)
        ax2.set_ylabel('Y', fontsize=10)
        ax2.set_aspect('equal')
        ax2.set_xlim(-1.1, 1.1)
        ax2.set_ylim(-1.1, 1.1)
        
        circle = Circle((0, 0), 1, fill=False, color='white', linewidth=2)
        ax2.add_patch(circle)
        
        # Plot 3: Line profile
        ax3.plot(v, line_profile, 'b-', linewidth=2, label='With spot')
        ax3.set_xlabel('Velocity (km/s)', fontsize=12)
        ax3.set_ylabel('Normalized Intensity', fontsize=12)
        ax3.set_title('Spectral Line Profile', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_ylim(0, 1.1)
        
        # Plot 4: Line profile comparison
        intensity_uniform = np.ones_like(intensity) * self._compute_limb_darkening(theta)
        v_uniform, line_uniform = self.compute_line_profile(theta, phi, intensity_uniform)
        
        ax4.plot(v, line_profile, 'b-', linewidth=2, label='With spot')
        ax4.plot(v_uniform, line_uniform, 'r--', linewidth=2, label='No spot')
        ax4.set_xlabel('Velocity (km/s)', fontsize=12)
        ax4.set_ylabel('Normalized Intensity', fontsize=12)
        ax4.set_title('Line Profile Comparison', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_ylim(0, 1.1)
        
        plt.suptitle('Stellar Spot Modeling and Spectral Line Effects', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig


def main() -> None:
    """
    Main function to demonstrate spectral line modeling with surface features.
    """
    print("Spectral Line Modeling with Stellar Surface Features")
    print("=" * 55)
    
    # Create modeler
    modeler = SpectralLineModeler(v_rot=50.0, v_micro=2.0, v_macro=5.0)
    
    # Demonstrate spherical harmonic features
    print("\n1. Modeling Y_2^0 features (zonal pattern)...")
    fig1 = modeler.plot_surface_and_line(l=2, m=0, feature_amplitude=0.4)
    plt.savefig('spectral_line_Y20.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n2. Modeling Y_3^1 features (tesseral pattern)...")
    fig2 = modeler.plot_surface_and_line(l=3, m=1, feature_amplitude=0.3)
    plt.savefig('spectral_line_Y31.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n3. Modeling stellar spot...")
    fig3 = modeler.plot_spot_model(spot_lat=30.0, spot_lon=0.0, 
                                  spot_radius=25.0, spot_temperature=0.6)
    plt.savefig('spectral_line_spot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n4. Modeling spot at different longitude...")
    fig4 = modeler.plot_spot_model(spot_lat=45.0, spot_lon=90.0, 
                                  spot_radius=20.0, spot_temperature=0.5)
    plt.savefig('spectral_line_spot_rotated.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nSpectral line modeling demonstration complete!")
    print("Check the generated PNG files for the visualizations.")


if __name__ == "__main__":
    main()
