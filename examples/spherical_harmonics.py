"""
Spherical Harmonics Visualization

A class for generating and plotting the first 20 spherical harmonics
with spherical projection using matplotlib and scipy.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm
from typing import Tuple, List, Optional
import warnings

# Suppress scipy warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


class SphericalHarmonicsPlotter:
    """
    A class for generating and visualizing spherical harmonics.
    
    This class provides methods to compute and plot the first 20 spherical
    harmonics (l=0 to l=4, m=-l to l) with proper spherical projection
    and color mapping.
    """
    
    def __init__(self, resolution: int = 100) -> None:
        """
        Initialize the spherical harmonics plotter.
        
        Parameters
        ----------
        resolution : int, optional
            Angular resolution for theta and phi grids, by default 100
        """
        self.resolution = resolution
        self.theta, self.phi = self._create_spherical_grid()
        self.harmonics_data: List[Tuple[int, int, np.ndarray]] = []
        
    def _create_spherical_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create spherical coordinate grids.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            theta and phi coordinate arrays
        """
        theta = np.linspace(0, np.pi, self.resolution)
        phi = np.linspace(0, 2 * np.pi, self.resolution)
        return theta, phi
    
    def _compute_spherical_harmonic(self, l: int, m: int) -> np.ndarray:
        """
        Compute spherical harmonic Y_l^m(theta, phi).
        
        Parameters
        ----------
        l : int
            Degree of spherical harmonic
        m : int
            Order of spherical harmonic
            
        Returns
        -------
        np.ndarray
            Complex spherical harmonic values
        """
        theta_grid, phi_grid = np.meshgrid(self.theta, self.phi, indexing='ij')
        return sph_harm(m, l, phi_grid, theta_grid)
    
    def generate_harmonics(self, max_l: int = 4, max_harmonics: int = 20) -> None:
        """
        Generate the first 20 spherical harmonics (l=0 to l=4, limited to 20).
        
        Parameters
        ----------
        max_l : int, optional
            Maximum degree l, by default 4
        max_harmonics : int, optional
            Maximum number of harmonics to generate, by default 20
        """
        self.harmonics_data = []
        count = 0
        
        for l in range(max_l + 1):
            for m in range(-l, l + 1):
                if count >= max_harmonics:
                    break
                harmonic = self._compute_spherical_harmonic(l, m)
                self.harmonics_data.append((l, m, harmonic))
                count += 1
            if count >= max_harmonics:
                break
    
    def _spherical_to_cartesian(self, r: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert spherical coordinates to Cartesian for plotting.
        
        Parameters
        ----------
        r : float, optional
            Radius of sphere, by default 1.0
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            x, y, z coordinate arrays
        """
        theta_grid, phi_grid = np.meshgrid(self.theta, self.phi, indexing='ij')
        
        x = r * np.sin(theta_grid) * np.cos(phi_grid)
        y = r * np.sin(theta_grid) * np.sin(phi_grid)
        z = r * np.cos(theta_grid)
        
        return x, y, z
    
    def plot_harmonics(self, 
                      figsize: Tuple[int, int] = (16, 12),
                      colormap: str = 'inferno',
                      alpha: float = 0.8,
                      show_phase: bool = True,
                      remove_axes: bool = True) -> plt.Figure:
        """
        Plot the first 20 spherical harmonics in a 4x5 grid.
        
        Parameters
        ----------
        figsize : Tuple[int, int], optional
            Figure size in inches, by default (16, 12)
        colormap : str, optional
            Matplotlib colormap name, by default 'inferno'
        alpha : float, optional
            Transparency of surface plots, by default 0.8
        show_phase : bool, optional
            Whether to show phase (imaginary part), by default True
        remove_axes : bool, optional
            Whether to remove the axes, by default True
        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        if not self.harmonics_data:
            self.generate_harmonics()
        
        fig = plt.figure(figsize=figsize)
        fig.suptitle('First 20 Spherical Harmonics Y_l^m(θ,φ)', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Create 4x5 subplot grid
        rows, cols = 4, 5
        
        for idx, (l, m, harmonic) in enumerate(self.harmonics_data):
            ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
            
            # Convert to Cartesian coordinates
            x, y, z = self._spherical_to_cartesian()
            
            # Choose real or imaginary part based on show_phase
            if show_phase:
                values = np.real(harmonic)
                title_suffix = "Re"
            else:
                values = np.imag(harmonic)
                title_suffix = "Im"
            
            # Create surface plot
            cmap = plt.colormaps[colormap]
            surf = ax.plot_surface(x, y, z, facecolors=cmap(
                Normalize(vmin=values.min(), vmax=values.max())(values)
            ), alpha=alpha, linewidth=0, antialiased=True)
            
            # Set equal aspect ratio
            ax.set_box_aspect([1, 1, 1])
            
            # Remove axes for cleaner look
            if remove_axes:
                ax.set_axis_off()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_zlabel('')
            
            # Set title
            ax.set_title(f'Y_{l}^{m}\n({title_suffix})', fontsize=10, pad=10)
            
            # Set viewing angle for better visualization
            ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        return fig
    
    def plot_single_harmonic(self, 
                           l: int, 
                           m: int,
                           figsize: Tuple[int, int] = (10, 8),
                           colormap: str = 'inferno',
                           show_phase: bool = True) -> plt.Figure:
        """
        Plot a single spherical harmonic with detailed visualization.
        
        Parameters
        ----------
        l : int
            Degree of spherical harmonic
        m : int
            Order of spherical harmonic
        figsize : Tuple[int, int], optional
            Figure size in inches, by default (10, 8)
        colormap : str, optional
            Matplotlib colormap name, by default 'inferno'
        show_phase : bool, optional
            Whether to show phase (imaginary part), by default True
            
        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        harmonic = self._compute_spherical_harmonic(l, m)
        x, y, z = self._spherical_to_cartesian()
        
        if show_phase:
            values = np.real(harmonic)
            title_suffix = "Real Part"
        else:
            values = np.imag(harmonic)
            title_suffix = "Imaginary Part"
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create surface plot
        cmap = plt.colormaps[colormap]
        surf = ax.plot_surface(x, y, z, facecolors=cmap(
            Normalize(vmin=values.min(), vmax=values.max())(values)
        ), alpha=0.8, linewidth=0, antialiased=True)
        
        # Add colorbar
        mappable = plt.cm.ScalarMappable(
            cmap=cmap, 
            norm=Normalize(vmin=values.min(), vmax=values.max())
        )
        mappable.set_array(values)
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label(f'Y_{l}^{m} {title_suffix}', fontsize=12)
        
        # Set labels and title
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title(f'Spherical Harmonic Y_{l}^{m}(θ,φ) - {title_suffix}', 
                    fontsize=14, fontweight='bold')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Set viewing angle
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        return fig
    
    def get_intensity_map(self, 
                         l: int, 
                         m: int,
                         r_max: float = 1.0,
                         resolution: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate 2D intensity map in polar coordinates (r, theta) for stellar disc.
        
        This function projects spherical harmonics onto a stellar disc using
        the relationship: r = sin(theta) for the stellar surface projection.
        
        Parameters
        ----------
        l : int
            Degree of spherical harmonic
        m : int
            Order of spherical harmonic
        r_max : float, optional
            Maximum radius of stellar disc, by default 1.0
        resolution : int, optional
            Resolution of the polar grid, by default 100
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            r coordinates, theta coordinates, and intensity values
        """
        # Create polar coordinate grid
        r = np.linspace(0, r_max, resolution)
        theta = np.linspace(0, 2 * np.pi, resolution)
        r_grid, theta_grid = np.meshgrid(r, theta, indexing='ij')
        
        # Convert to spherical coordinates for harmonic calculation
        # For stellar disc projection: r = sin(theta_spherical)
        # So theta_spherical = arcsin(r), phi = theta_polar
        theta_spherical = np.arcsin(np.clip(r_grid, 0, 1))
        phi_spherical = theta_grid
        
        # Compute spherical harmonic
        harmonic = sph_harm(m, l, phi_spherical, theta_spherical)
        
        # Take real part for intensity map
        intensity = np.real(harmonic)
        
        return r_grid, theta_grid, intensity
    
    def plot_intensity_map(self, 
                          l: int, 
                          m: int,
                          r_max: float = 1.0,
                          figsize: Tuple[int, int] = (10, 8),
                          colormap: str = 'inferno') -> plt.Figure:
        """
        Plot intensity map in polar coordinates for stellar disc.
        
        Parameters
        ----------
        l : int
            Degree of spherical harmonic
        m : int
            Order of spherical harmonic
        r_max : float, optional
            Maximum radius of stellar disc, by default 1.0
        figsize : Tuple[int, int], optional
            Figure size in inches, by default (10, 8)
        colormap : str, optional
            Matplotlib colormap name, by default 'inferno'
            
        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        r_grid, theta_grid, intensity = self.get_intensity_map(l, m, r_max)
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        # Create polar plot
        im = ax.pcolormesh(theta_grid, r_grid, intensity, cmap=colormap, shading='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.1)
        cbar.set_label(f'Intensity Y_{l}^{m}', fontsize=12)
        
        # Set labels and title
        ax.set_title(f'Stellar Disc Intensity Map Y_{l}^{m}(r,θ)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Azimuthal Angle θ', fontsize=12)
        ax.set_ylabel('Radial Distance r', fontsize=12)
        
        # Set radial limits
        ax.set_ylim(0, r_max)
        
        plt.tight_layout()
        return fig

    def plot_2d_projection(self, 
                          l: int, 
                          m: int,
                          figsize: Tuple[int, int] = (12, 5),
                          colormap: str = 'inferno') -> plt.Figure:
        """
        Plot 2D projections of a spherical harmonic.
        
        Parameters
        ----------
        l : int
            Degree of spherical harmonic
        m : int
            Order of spherical harmonic
        figsize : Tuple[int, int], optional
            Figure size in inches, by default (12, 5)
        colormap : str, optional
            Matplotlib colormap name, by default 'inferno'
            
        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        harmonic = self._compute_spherical_harmonic(l, m)
        theta_grid, phi_grid = np.meshgrid(self.theta, self.phi, indexing='ij')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Real part
        real_values = np.real(harmonic)
        im1 = ax1.imshow(real_values, extent=[0, 2*np.pi, 0, np.pi], 
                        aspect='auto', cmap=colormap, origin='lower')
        ax1.set_xlabel('φ (azimuthal angle)', fontsize=12)
        ax1.set_ylabel('θ (polar angle)', fontsize=12)
        ax1.set_title(f'Real Part of Y_{l}^{m}', fontsize=12, fontweight='bold')
        ax1.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax1.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        ax1.set_yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        ax1.set_yticklabels(['0', 'π/4', 'π/2', '3π/4', 'π'])
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # Imaginary part
        imag_values = np.imag(harmonic)
        im2 = ax2.imshow(imag_values, extent=[0, 2*np.pi, 0, np.pi], 
                        aspect='auto', cmap=colormap, origin='lower')
        ax2.set_xlabel('φ (azimuthal angle)', fontsize=12)
        ax2.set_ylabel('θ (polar angle)', fontsize=12)
        ax2.set_title(f'Imaginary Part of Y_{l}^{m}', fontsize=12, fontweight='bold')
        ax2.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax2.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        ax2.set_yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        ax2.set_yticklabels(['0', 'π/4', 'π/2', '3π/4', 'π'])
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        plt.suptitle(f'2D Projections of Spherical Harmonic Y_{l}^{m}(θ,φ)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig


def main() -> None:
    """
    Main function to demonstrate the SphericalHarmonicsPlotter class.
    """
    # Create plotter instance
    plotter = SphericalHarmonicsPlotter(resolution=80)
    
    # Generate and plot all harmonics
    print("Generating first 20 spherical harmonics...")
    fig_all = plotter.plot_harmonics(colormap='inferno')
    plt.show()
    
    # Plot a specific harmonic in detail
    print("Plotting Y_3^1 in detail...")
    fig_single = plotter.plot_single_harmonic(l=3, m=1, colormap='inferno')
    plt.show()
    
    # Plot 2D projections
    print("Plotting 2D projections of Y_2^0...")
    fig_2d = plotter.plot_2d_projection(l=2, m=0, colormap='inferno')
    plt.show()
    
    print("Spherical harmonics visualization complete!")


if __name__ == "__main__":
    main()
