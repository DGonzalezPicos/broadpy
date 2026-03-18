import numpy as np
import matplotlib.pyplot as plt

def get_velocity_field(wave, spec, vsini, nr=10, ntheta=30, eps=0.6):
    dr = 1./nr
    r = 0.5 * dr + dr * np.arange(0, nr)  # Radial values from 0 to the radius
    theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)  # Azimuthal values from 0 to 2*pi
    R, Theta = np.meshgrid(r, theta)  # Create a grid of radial and azimuthal values
    v_proj = vsini * R * np.sin(Theta)
    return v_proj, R, Theta

def spherical_harmonics(theta, phi, l, m):
    """
    Compute spherical harmonics Y_l^m(theta, phi) for given degree l and order m.
    
    Parameters
    ----------
    theta : array_like
        Colatitude (0 to π)
    phi : array_like  
        Azimuthal angle (0 to 2π)
    l : int
        Degree of spherical harmonic
    m : int
        Order of spherical harmonic (-l <= m <= l)
        
    Returns
    -------
    Y : array_like
        Spherical harmonic values
    """
    from scipy.special import lpmv, factorial
    
    # Normalization factor
    N = np.sqrt((2*l + 1) / (4*np.pi) * factorial(l - abs(m)) / factorial(l + abs(m)))
    
    # Associated Legendre polynomial
    P = lpmv(abs(m), l, np.cos(theta))
    
    # Spherical harmonic
    if m >= 0:
        Y = N * P * np.cos(m * phi)
    else:
        Y = N * P * np.sin(abs(m) * phi)
    
    return Y


def brightness_map_spherical_harmonics(R, Theta, Y_coeffs=None, ydeg=5):
    """
    Generate a brightness map using spherical harmonics, similar to starry package.
    
    Parameters
    ----------
    R : array_like
        Radial coordinates (0 to 1)
    Theta : array_like
        Azimuthal coordinates (0 to 2π)
    Y_coeffs : array_like, optional
        Spherical harmonic coefficients. If None, uses default coefficients.
        Indexed as [Y_0,0, Y_1,-1, Y_1,0, Y_1,1, Y_2,-2, ...]
    ydeg : int, optional
        Maximum degree of spherical harmonics (default: 5)
        
    Returns
    -------
    brightness : array_like
        Brightness map values
    """
    # Convert to spherical coordinates
    # R is normalized radius (0 to 1), Theta is azimuthal angle (0 to 2π)
    # For a disk, we use colatitude theta = π/2 (equatorial plane)
    theta = np.full_like(R, np.pi/2)  # Colatitude = π/2 for disk
    phi = Theta  # Azimuthal angle
    
    # Default coefficients (similar to starry's Earth map example)
    if Y_coeffs is None:
        Y_coeffs = np.array([
            1.00,   # Y_0,0
            0.22, 0.19, 0.11,  # Y_1,-1, Y_1,0, Y_1,1
            0.11, 0.07, -0.11, 0.00, -0.05,  # Y_2,-2 to Y_2,2
            0.12, 0.16, -0.05, 0.06, 0.12, 0.05, -0.10,  # Y_3,-3 to Y_3,3
            0.04, -0.02, 0.01, 0.10, 0.08, 0.15, 0.13, -0.11, -0.07,  # Y_4,-4 to Y_4,4
            -0.14, 0.06, -0.19, -0.02, 0.07, -0.02, 0.07, -0.01, -0.07, 0.04, 0.00  # Y_5,-5 to Y_5,5
        ])
    
    # Ensure we have enough coefficients for the requested degree
    n_coeffs_needed = (ydeg + 1)**2
    if len(Y_coeffs) < n_coeffs_needed:
        # Pad with zeros
        Y_coeffs = np.pad(Y_coeffs, (0, n_coeffs_needed - len(Y_coeffs)), 'constant')
    elif len(Y_coeffs) > n_coeffs_needed:
        # Truncate
        Y_coeffs = Y_coeffs[:n_coeffs_needed]
    
    # Initialize brightness map
    brightness = np.zeros_like(R)
    
    # Sum over all spherical harmonics
    idx = 0
    for l in range(ydeg + 1):
        for m in range(-l, l + 1):
            if idx < len(Y_coeffs):
                Y_lm = spherical_harmonics(theta, phi, l, m)
                brightness += Y_coeffs[idx] * Y_lm
                idx += 1
    
    # Ensure non-negative brightness (physical constraint)
    brightness = np.maximum(brightness, 0)
    
    return brightness


def create_starry_like_map(ydeg=5, Y_coeffs=None):
    """
    Create a starry-like map object with spherical harmonic coefficients.
    
    Parameters
    ----------
    ydeg : int, optional
        Maximum degree of spherical harmonics (default: 5)
    Y_coeffs : array_like, optional
        Spherical harmonic coefficients
        
    Returns
    -------
    map_dict : dict
        Dictionary containing map information similar to starry.Map
    """
    # Default coefficients (Earth-like map)
    if Y_coeffs is None:
        Y_coeffs = np.array([
            1.00,   # Y_0,0
            0.22, 0.19, 0.11,  # Y_1,-1, Y_1,0, Y_1,1
            0.11, 0.07, -0.11, 0.00, -0.05,  # Y_2,-2 to Y_2,2
            0.12, 0.16, -0.05, 0.06, 0.12, 0.05, -0.10,  # Y_3,-3 to Y_3,3
            0.04, -0.02, 0.01, 0.10, 0.08, 0.15, 0.13, -0.11, -0.07,  # Y_4,-4 to Y_4,4
            -0.14, 0.06, -0.19, -0.02, 0.07, -0.02, 0.07, -0.01, -0.07, 0.04, 0.00  # Y_5,-5 to Y_5,5
        ])
    
    # Create map dictionary
    map_dict = {
        'ydeg': ydeg,
        'y': Y_coeffs,
        'n_coeffs': len(Y_coeffs)
    }
    
    return map_dict


def render_map(map_dict, resolution=100):
    """
    Render a spherical harmonic map to a 2D image, similar to starry's show() method.
    
    Parameters
    ----------
    map_dict : dict
        Map dictionary from create_starry_like_map()
    resolution : int, optional
        Resolution of the rendered image (default: 100)
        
    Returns
    -------
    image : array_like
        2D array representing the rendered map
    x : array_like
        x coordinates
    y : array_like
        y coordinates
    """
    # Create coordinate grid
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Convert to polar coordinates
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)
    
    # Mask points outside the unit circle
    mask = R <= 1.0
    
    # Generate brightness map
    brightness = brightness_map_spherical_harmonics(
        R, Theta, 
        Y_coeffs=map_dict['y'], 
        ydeg=map_dict['ydeg']
    )
    
    # Apply mask
    brightness[~mask] = 0
    
    return brightness, x, y



if __name__ == '__main__':
    
    from broadpy.utils import load_example_data

    # Example 1: Original velocity field visualization
    print("Example 1: Velocity Field Visualization")
    wave, spec = load_example_data(wave_range=(2322, 2324)) 
    nr = 10
    ntheta = 30
    vsini = 10.0
    eps = 0.0
    
    v_proj, R, Theta = get_velocity_field(wave, spec, vsini, nr, ntheta, eps)
    
    # Plot the velocity field in polar coordinates
    fig = plt.figure(tight_layout=True, figsize=(15, 10))
    
    # Velocity field plot
    ax1 = fig.add_subplot(2, 2, 1, projection='polar')
    ax1.grid(False)
    cax1 = ax1.pcolormesh(Theta+np.pi/2, R, -v_proj, cmap='bwr')
    fig.colorbar(cax1, ax=ax1, label='Projected Velocity (km/s)', orientation='horizontal', pad=0.1)
    ax1.set_title('Velocity Field')
    
    # Example 2: Spherical harmonics brightness map
    print("Example 2: Spherical Harmonics Brightness Map")
    
    # Create a starry-like map
    map_dict = create_starry_like_map(ydeg=5)
    print(f"Created map with {map_dict['n_coeffs']} spherical harmonic coefficients")
    print(f"Y_0,0 coefficient: {map_dict['y'][0]:.3f}")
    
    # Render the map
    brightness, x, y = render_map(map_dict, resolution=100)
    
    # Plot the brightness map
    ax2 = fig.add_subplot(2, 2, 2)
    im2 = ax2.imshow(brightness, extent=[-1, 1, -1, 1], origin='lower', cmap='plasma')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Spherical Harmonics Map (Earth-like)')
    fig.colorbar(im2, ax=ax2, label='Brightness')
    
    # Example 3: Custom spherical harmonics map
    print("Example 3: Custom Spherical Harmonics Map")
    
    # Create custom coefficients for a different pattern
    custom_coeffs = np.zeros(36)  # (5+1)^2 = 36 coefficients
    custom_coeffs[0] = 1.0        # Y_0,0 (constant)
    custom_coeffs[1] = 0.5        # Y_1,-1
    custom_coeffs[3] = -0.3       # Y_1,1
    custom_coeffs[4] = 0.2        # Y_2,-2
    custom_coeffs[8] = -0.4       # Y_2,2
    
    custom_map = create_starry_like_map(ydeg=5, Y_coeffs=custom_coeffs)
    custom_brightness, _, _ = render_map(custom_map, resolution=100)
    
    # Plot custom map
    ax3 = fig.add_subplot(2, 2, 3)
    im3 = ax3.imshow(custom_brightness, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Custom Spherical Harmonics Map')
    fig.colorbar(im3, ax=ax3, label='Brightness')
    
    # Example 4: Individual spherical harmonics
    print("Example 4: Individual Spherical Harmonics")
    
    # Create coordinate grid for individual harmonics
    x_harm = np.linspace(-1, 1, 50)
    y_harm = np.linspace(-1, 1, 50)
    X_harm, Y_harm = np.meshgrid(x_harm, y_harm)
    R_harm = np.sqrt(X_harm**2 + Y_harm**2)
    Theta_harm = np.arctan2(Y_harm, X_harm)
    
    # Show Y_2,0 (zonal harmonic)
    Y_20 = brightness_map_spherical_harmonics(R_harm, Theta_harm, 
                                            Y_coeffs=np.array([0, 0, 0, 0, 0, 0, 1, 0, 0]), 
                                            ydeg=2)
    Y_20[R_harm > 1] = 0  # Mask outside unit circle
    
    ax4 = fig.add_subplot(2, 2, 4)
    im4 = ax4.imshow(Y_20, extent=[-1, 1, -1, 1], origin='lower', cmap='RdBu_r')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title('Y₂,₀ Spherical Harmonic')
    fig.colorbar(im4, ax=ax4, label='Amplitude')
    
    plt.tight_layout()
    plt.show()
    
    # Print some information about the spherical harmonics
    print("\nSpherical Harmonics Information:")
    print(f"Map degree: {map_dict['ydeg']}")
    print(f"Number of coefficients: {map_dict['n_coeffs']}")
    print(f"First few coefficients: {map_dict['y'][:10]}")
    
    # Demonstrate coefficient indexing (similar to starry)
    print("\nCoefficient indexing (similar to starry):")
    idx = 0
    for l in range(min(3, map_dict['ydeg'] + 1)):
        for m in range(-l, l + 1):
            if idx < len(map_dict['y']):
                print(f"Y_{l},{m:2d} = {map_dict['y'][idx]:8.3f}")
                idx += 1