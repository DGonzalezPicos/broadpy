"""
Gaussian Absorption Line Modeling with Stellar Surface Features

This script demonstrates how different stellar surface brightness maps affect
a single Gaussian absorption line, using velocity fields and spherical harmonics.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.special import voigt_profile
from typing import Tuple, List, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Import our spherical harmonics class
from spherical_harmonics import SphericalHarmonicsPlotter


class GaussianLineModeler:
    """
    A class for modeling Gaussian absorption lines with stellar surface features.
    
    This class combines velocity fields, brightness maps, and spherical harmonics
    to simulate how stellar surface features affect Gaussian absorption lines.
    """
    
    def __init__(self, 
                 vsini: float = 20.0,
                 resolution: int = 100) -> None:
        """
        Initialize the Gaussian line modeler.
        
        Parameters
        ----------
        vsini : float, optional
            Stellar rotation velocity in km/s, by default 20.0
        resolution : int, optional
            Angular resolution for surface grid, by default 100
        """
        self.vsini = vsini
        self.resolution = resolution
        
        # Create surface coordinate grids
        self.r, self.theta = self._create_surface_grid()
        self.R, self.Theta = np.meshgrid(self.r, self.theta)
        
        # Initialize spherical harmonics plotter
        self.harmonics_plotter = SphericalHarmonicsPlotter(resolution=resolution)
    
    def _create_surface_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create surface coordinate grids for stellar surface.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            r and theta coordinate arrays
        """
        # Use the same grid as velocity_brightness_field.py
        nr = self.resolution // 10  # Adjust resolution for performance
        ntheta = self.resolution
        
        dr = 1.0 / nr
        r = 0.5 * dr + dr * np.arange(0, nr)  # Radial values from 0 to the radius
        # Adjust theta so rotation axis points north (up)
        # Start from -π/2 to get blueshifted side on left, redshifted on right
        theta = np.linspace(-np.pi/2, 3*np.pi/2, ntheta, endpoint=False)
        
        return r, theta
    
    def get_velocity_field(self) -> np.ndarray:
        """
        Get velocity field using the same method as velocity_brightness_field.py.
        
        Returns
        -------
        np.ndarray
            Projected velocity field
        """
        # Use the same formula as velocity_brightness_field.py
        # With corrected orientation: blueshifted (negative) on left, redshifted (positive) on right
        v_proj = self.vsini * self.R * np.cos(self.Theta)
        return v_proj
    
    def create_uniform_brightness(self) -> np.ndarray:
        """
        Create uniform brightness map.
        
        Returns
        -------
        np.ndarray
            Uniform brightness array
        """
        return np.ones_like(self.R)
    
    def create_limb_darkening_brightness(self, u: float = 0.6) -> np.ndarray:
        """
        Create limb darkening brightness map using spherical harmonics.
        
        Parameters
        ----------
        u : float, optional
            Limb darkening coefficient, by default 0.6
            
        Returns
        -------
        np.ndarray
            Limb darkening brightness array
        """
        # Convert to spherical coordinates for limb darkening
        # For stellar disc: r = sin(theta_spherical)
        # So theta_spherical = arcsin(r), and we need cos(theta_spherical)
        theta_spherical = np.arcsin(np.clip(self.R, 0, 1))
        mu = np.cos(theta_spherical)
        
        # Linear limb darkening: I = I0 * (1 - u + u*mu)
        brightness = 1 - u + u * mu
        
        return brightness
    
    def create_spherical_harmonic_brightness(self, 
                                           l: int, 
                                           m: int,
                                           amplitude: float = 0.3) -> np.ndarray:
        """
        Create brightness map using spherical harmonics.
        
        Parameters
        ----------
        l : int
            Degree of spherical harmonic
        m : int
            Order of spherical harmonic
        amplitude : float, optional
            Amplitude of the harmonic feature, by default 0.3
            
        Returns
        -------
        np.ndarray
            Spherical harmonic brightness array
        """
        # Convert to spherical coordinates for harmonic calculation
        theta_spherical = np.arcsin(np.clip(self.R, 0, 1))
        # Adjust phi to match the corrected theta coordinate system
        phi_spherical = self.Theta + np.pi/2  # Rotate by 90 degrees to match orientation
        
        # Compute spherical harmonic directly for this grid
        from scipy.special import sph_harm
        harmonic = sph_harm(m, l, phi_spherical, theta_spherical)
        harmonic_real = np.real(harmonic)
        
        # Normalize and scale
        harmonic_norm = harmonic_real / np.max(np.abs(harmonic_real))
        brightness = 1.0 + amplitude * harmonic_norm
        
        # Apply basic limb darkening
        mu = np.cos(theta_spherical)
        limb_darkening = 1 - 0.6 + 0.6 * mu
        brightness *= limb_darkening
        
        return brightness
    
    def create_spot_brightness(self, 
                              spot_lat: float = 30.0,
                              spot_lon: float = 0.0,
                              spot_radius: float = 20.0,
                              spot_temperature: float = 0.7) -> np.ndarray:
        """
        Create spot brightness map.
        
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
        np.ndarray
            Spot brightness array
        """
        # Convert to degrees, adjusting for the corrected coordinate system
        theta_deg = np.degrees(self.Theta + np.pi/2)  # Adjust for orientation
        r_deg = np.degrees(np.arcsin(np.clip(self.R, 0, 1)))
        
        # Create spot
        spot_lat_rad = np.radians(spot_lat)
        spot_lon_rad = np.radians(spot_lon)
        spot_radius_rad = np.radians(spot_radius)
        
        # Compute angular distance from spot center
        cos_distance = (np.sin(np.radians(r_deg)) * np.sin(spot_lat_rad) + 
                       np.cos(np.radians(r_deg)) * np.cos(spot_lat_rad) * 
                       np.cos(np.radians(theta_deg) - spot_lon_rad))
        
        # Create spot mask
        spot_mask = cos_distance > np.cos(spot_radius_rad)
        
        # Create brightness map
        brightness = np.ones_like(self.R)
        brightness[spot_mask] = spot_temperature
        
        # Apply limb darkening
        theta_spherical = np.arcsin(np.clip(self.R, 0, 1))
        mu = np.cos(theta_spherical)
        limb_darkening = 1 - 0.6 + 0.6 * mu
        brightness *= limb_darkening
        
        return brightness
    
    def compute_gaussian_line_profile(self, 
                                    brightness: np.ndarray,
                                    line_center: float = 0.0,
                                    line_width: float = 1.0,
                                    line_depth: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Gaussian absorption line profile.
        
        Parameters
        ----------
        brightness : np.ndarray
            Surface brightness array
        line_center : float, optional
            Central velocity in km/s, by default 0.0
        line_width : float, optional
            Gaussian width parameter, by default 1.0
        line_depth : float, optional
            Maximum line depth, by default 0.8
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Velocity array and line profile
        """
        # Create velocity grid
        v_max = self.vsini + 5  # Add small buffer
        v = np.linspace(-v_max, v_max, 300)
        
        # Get velocity field
        v_doppler = self.get_velocity_field()
        
        # Initialize line profile (continuum = 1)
        line_profile = np.ones_like(v)
        
        # Compute total stellar flux for normalization
        total_flux = 0.0
        for i in range(self.R.shape[0]):
            for j in range(self.R.shape[1]):
                # Skip elements behind the star (r > 1)
                if self.R[i, j] > 1.0:
                    continue
                
                # Compute element area (solid angle)
                dr = 1.0 / (self.resolution // 10)
                dtheta = 2 * np.pi / self.resolution
                domega = self.R[i, j] * dr * dtheta
                
                # Add to total flux
                total_flux += brightness[i, j] * domega
        
        # Add absorption contribution from each surface element
        for i in range(self.R.shape[0]):
            for j in range(self.R.shape[1]):
                # Skip elements behind the star (r > 1)
                if self.R[i, j] > 1.0:
                    continue
                
                # Compute element area (solid angle)
                dr = 1.0 / (self.resolution // 10)
                dtheta = 2 * np.pi / self.resolution
                domega = self.R[i, j] * dr * dtheta
                
                # Compute element contribution
                v_element = line_center + v_doppler[i, j]
                element_brightness = brightness[i, j] * domega
                
                # Add Gaussian absorption contribution
                for k, v_val in enumerate(v):
                    gaussian = np.exp(-0.5 * ((v_val - v_element) / line_width)**2)
                    # Scale absorption by element brightness and line depth
                    absorption = line_depth * gaussian * (element_brightness / total_flux)
                    line_profile[k] -= absorption
        
        # Ensure physical values: continuum = 1, line depth between 0 and 1
        line_profile = np.clip(line_profile, 0, 1)
        
        return v, line_profile
    
    def plot_simple_comparison(self, 
                              brightness_maps: List[Tuple[np.ndarray, str]],
                              figsize: Tuple[int, int] = (16, 12),
                              vmin=None,
                              vmax=None) -> plt.Figure:
        """
        Plot simple comparison with velocity map, brightness maps, overlapping line profiles, and residuals.
        
        Parameters
        ----------
        brightness_maps : List[Tuple[np.ndarray, str]]
            List of (brightness_array, title) tuples
        figsize : Tuple[int, int], optional
            Figure size in inches, by default (16, 12)
        vmin : float, optional
            Minimum value for brightness maps, by default None (between 0 and 1)
        vmax : float, optional
            Maximum value for brightness maps, by default None
        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        n_maps = len(brightness_maps)
        
        # Create figure with proper layout using GridSpec for height control
        from matplotlib.gridspec import GridSpec
        fig = plt.figure(figsize=figsize)
        
        # Create GridSpec with 3 rows and n_maps+1 columns
        gs = GridSpec(3, n_maps + 1, height_ratios=[2, 3, 2], hspace=0.3, wspace=0.3)
        
        # Create polar subplots for maps (top row)
        polar_axes = []
        for i in range(n_maps + 1):
            ax = fig.add_subplot(gs[0, i], projection='polar')
            polar_axes.append(ax)
        
        # Create Cartesian subplot for line profiles (middle row, spanning all columns)
        ax_line = fig.add_subplot(gs[1, :])
        
        # Create Cartesian subplot for residuals (bottom row, spanning all columns)
        ax_residuals = fig.add_subplot(gs[2, :])
        # Plot velocity map
        v_doppler = self.get_velocity_field()
        im0 = polar_axes[0].pcolormesh(self.Theta, self.R, v_doppler, 
                                      cmap='RdBu_r', shading='auto')
        polar_axes[0].set_title('Velocity Field\n(km/s)', fontsize=12, fontweight='bold', pad=20)
        polar_axes[0].set_ylim(0, 1)
        polar_axes[0].set_xticks([])
        polar_axes[0].set_yticks([])
        plt.colorbar(im0, ax=polar_axes[0], shrink=0.8, label='Velocity (km/s)')
        
        # Plot brightness maps
        for i, (brightness, title) in enumerate(brightness_maps):
            im = polar_axes[i + 1].pcolormesh(self.Theta, self.R, brightness, 
                                             cmap='inferno', shading='auto', vmin=vmin, vmax=vmax)
            polar_axes[i + 1].set_title(f'{title}\nBrightness', fontsize=12, fontweight='bold', pad=20)
            polar_axes[i + 1].set_ylim(0, 1)
            polar_axes[i + 1].set_xticks([])
            polar_axes[i + 1].set_yticks([])
            plt.colorbar(im, ax=polar_axes[i + 1], shrink=0.8, label='Brightness')
        
        # Plot overlapping line profiles in Cartesian coordinates
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        line_profiles = []
        
        for i, (brightness, title) in enumerate(brightness_maps):
            v, line_profile = self.compute_gaussian_line_profile(brightness)
            line_profiles.append((v, line_profile, title))
            ax_line.plot(v, line_profile, color=colors[i % len(colors)], 
                        linewidth=2, label=title)
        
        # Plot residuals with respect to the first line profile
        if len(line_profiles) > 1:
            reference_v, reference_profile, reference_title = line_profiles[0]
            for i, (v, line_profile, title) in enumerate(line_profiles[1:], 1):
                # Interpolate to match reference velocity grid
                residual = np.interp(reference_v, v, line_profile) - reference_profile
                ax_residuals.plot(reference_v, residual, color=colors[i % len(colors)], 
                                linewidth=2, label=f'{title} - {reference_title}')
        
        # Set labels and formatting for line profiles
        ax_line.set_xlabel('Radial Velocity (km/s)', fontsize=12)
        ax_line.set_ylabel('Normalized Intensity', fontsize=12)
        ax_line.set_title('Gaussian Line Profiles (All Overlaid)', fontsize=12, fontweight='bold')
        ax_line.grid(True, alpha=0.3)
        ax_line.legend()
        
        # Set labels and formatting for residuals
        ax_residuals.set_xlabel('Radial Velocity (km/s)', fontsize=12)
        ax_residuals.set_ylabel('Residual Intensity', fontsize=12)
        ax_residuals.set_title('Residuals (Relative to Uniform Brightness)', fontsize=12, fontweight='bold')
        ax_residuals.grid(True, alpha=0.3)
        ax_residuals.legend()
        ax_residuals.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # plt.suptitle('Gaussian Absorption Line Modeling with Stellar Surface Features', 
        #             fontsize=16, fontweight='bold')
        return fig


def main() -> None:
    """
    Main function to demonstrate Gaussian line modeling with different surface features.
    """
    print("Gaussian Absorption Line Modeling with Stellar Surface Features")
    print("=" * 65)
    
    # Create modeler with vsini = 20 km/s
    modeler = GaussianLineModeler(vsini=20.0)
    
    print(f"Modeling with vsini = {modeler.vsini} km/s")
    print(f"Surface grid: {modeler.R.shape[0]} × {modeler.R.shape[1]}")
    
    # Create different brightness maps
    print("\nCreating brightness maps...")
    uniform_brightness = modeler.create_uniform_brightness()
    limb_darkening_brightness = modeler.create_limb_darkening_brightness(u=0.6)
    harmonic_brightness = modeler.create_spherical_harmonic_brightness(l=2, m=0, amplitude=0.3)
    spot_brightness = modeler.create_spot_brightness(
        spot_lat=30.0, spot_lon=0.0, spot_radius=25.0, spot_temperature=0.6
    )
    
    # Create brightness maps list
    brightness_maps = [
        (uniform_brightness, "Uniform"),
        (limb_darkening_brightness, "Limb Darkening"),
        (harmonic_brightness, "Y₂⁰ Harmonic"),
        (spot_brightness, "Stellar Spot")
    ]
    
    print("Generating comparison plot...")
    
    # Create single comparison figure
    fig = modeler.plot_simple_comparison(brightness_maps)
    plt.savefig('gaussian_line_simple.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Gaussian line modeling demonstration complete!")
    print("Check 'gaussian_line_simple.png' for the visualization.")
    
    # Print quantitative results
    print("\nQuantitative Analysis:")
    for i, (brightness, title) in enumerate(brightness_maps):
        v, line_profile = modeler.compute_gaussian_line_profile(brightness)
        min_intensity = np.min(line_profile)
        print(f"{title}: Min intensity = {min_intensity:.3f}")


if __name__ == "__main__":
    main()
