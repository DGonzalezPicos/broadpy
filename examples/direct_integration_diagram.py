import numpy as np
import matplotlib.pyplot as plt

def get_velocity_field(wave, spec, vsini, nr=10, ntheta=30, eps=0.6):
    dr = 1./nr
    r = 0.5 * dr + dr * np.arange(0, nr)  # Radial values from 0 to the radius
    theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)  # Azimuthal values from 0 to 2*pi
    R, Theta = np.meshgrid(r, theta)  # Create a grid of radial and azimuthal values
    v_proj = vsini * R * np.sin(Theta)
    return v_proj, R, Theta


def direct_integration(wave, spec, vsini, nr=10, ntheta=30, eps=0.6, 
                       return_all=False):
    # nr = 10  # Number of radial points
    dr = 1./nr

    # ntheta = 30  # Number of azimuthal points
    # vsini = 10.0  # Projected rotational velocity in km/s

    r = 0.5 * dr + dr * np.arange(0, nr)  # Radial values from 0 to the radius
    theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)  # Azimuthal values from 0 to 2*pi
    # k = np.array([np.arange(0, rmax) for rmax in np.rint(ntheta*r).astype(int)])
    # theta = np.pi/np.rint(ntheta*r) + k * 2.0*np.pi/np.rint(ntheta*r)

    R, Theta = np.meshgrid(r, theta)  # Create a grid of radial and azimuthal values

    # Calculate projected velocity
    v_proj, R, Theta = get_velocity_field(wave, spec, vsini, nr, ntheta, eps)
    n_edge = np.ceil(v_proj.max() / 2.9979e5 * 1000)  # Number of edge points to remove
    # eps = 0.6 # Limb darkening coefficient

    # Calculate the area of each block with limb darkening

    area = np.pi * (R + 0.5*dr)**2 - np.pi * (R - 0.5*dr)**2
    area *= (1.0 - eps + eps*np.cos(np.arcsin(R)))
    total_area = area.sum()



    wave_sampling = np.mean(np.diff(wave))
    n_edge = 1 + int(np.ceil((np.mean(wave)/wave_sampling) * np.abs(v_proj.max()) / 2.99792458e5))


    from scipy.interpolate import interp1d

    inter = interp1d(wave, spec, kind='linear', bounds_error=False, fill_value=0.0)
    spec_shift = inter(np.outer(wave[n_edge:-n_edge], (1 + v_proj/2.9979e5)))
    spec_broad = spec_shift.dot(area.flatten()) / total_area
    if return_all:
        return spec_broad, n_edge, spec_shift, v_proj, R, Theta
    
    return spec_broad




if __name__ == '__main__':
    import matplotlib.gridspec as gridspec
    from broadpy.utils import load_example_data
    plt.style.use('dark_background')
    plt.rcParams['font.size'] = '16'
    


    wave, spec = load_example_data(wave_range=(2322, 2324))  

    vsini, nr, ntheta = 10.0, 10, 30
    eps = 0.0
    # spec_broad = direct_integration(wave, spec, vsini=vsini, nr=nr, ntheta=ntheta)
    spec_broad, n_edge, spec_shift, v_proj, R, Theta = direct_integration(wave, spec, 
                                                                        vsini=vsini, nr=nr, ntheta=ntheta, eps=eps, 
                                                                        return_all=True)
    
    # apply brightness map to spec_shift to simulate a dark spot on the surface
    

    # Plot the projected velocity in polar coordinates
    # generate two subplots, one
    ns = spec_shift.shape[1]
    
    # cmap = plt.cm.get_cmap('bwr_r', len(ns))
    cmap = plt.cm.get_cmap('bwr', ns)
    colors = [cmap(x) for x in range(ns)]
    fig = plt.figure(tight_layout=True, figsize=(12,7))
    gs = gridspec.GridSpec(2,3)
    ax0 = fig.add_subplot(gs[1, :2])
    ax1 = fig.add_subplot(gs[0, :2], sharex=ax0)
    ax2 = fig.add_subplot(gs[:,2], projection='polar')
    
    
    ax0.plot(wave, spec, c='w', alpha=0.4, label='Original')
    ax0.plot(wave[n_edge:-n_edge], spec_broad, label='Direct Integration', c='orange')
    # ax0.plot(wave, spec_rotbroad, label='rotbroad', ls='--', alpha=0.5)
    
    for i in range(ns):
        ax1.plot(wave[n_edge:-n_edge], spec_shift[:,i], color=colors[i], alpha=0.8)


    ax2.grid(False)  # Disable grid removal warning
    cax = ax2.pcolormesh(Theta+np.pi/2, R, -v_proj, cmap='bwr')
    fig.colorbar(cax, ax=ax2, label='Projected Velocity (km/s)', orientation='horizontal', pad=0.1)
    
    # Set plot properties
    ax0.set(xlabel='Wavelength (A)', ylabel='Flux [cgs]', xlim=np.percentile(wave, [30, 50]))
    ax0.legend()
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])

    plt.title('2D Object Surface')
    plt.ion()
    plt.show()
    fig.savefig('direct_integration_diagram.pdf', dpi=300, bbox_inches='tight')