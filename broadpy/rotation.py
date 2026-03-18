import numpy as np
from scipy.ndimage import convolve1d

class RotationalBroadening:
    
    c = 2.998e5  # Speed of light in km/s
    epsilon = 0.6  # Limb darkening coefficient, default: 0.6
    
    def __init__(self, wavelength, flux):
        self.wavelength = wavelength  # Wavelength array
        self.flux = flux  # Flux array
        assert len(self.wavelength) == len(self.flux), (
            '`wavelength` and `flux` should have the same length.'
        )
        assert not np.any(np.isnan(self.wavelength)), '`wavelength` should not contain NaN.'

        self.spacing = np.mean(
            2 * np.diff(self.wavelength) / (self.wavelength[1:] + self.wavelength[:-1])
        )
        
        # reference wavelength
        self.ref_wave = np.mean(self.wavelength)
        self.dw = np.mean(np.diff(self.wavelength))
        
    
    def __call__(self, vsini, epsilon=0.6):
        '''Apply rotational broadening to the spectral line
        
        Parameters
        ----------
        vsini : float
            Projected rotational velocity in km/s
        epsilon : float, optional
            Limb darkening coefficient (default: 0.6)
        
        Returns
        -------
        y_broadened : array
            Flux array after rotational broadening
        '''
        self.eps = epsilon
        _kernel = self.rotational_kernel(vsini, self.ref_wave, self.dw)
        y_broadened = convolve1d(self.flux, _kernel, mode='nearest') * self.dw
        return y_broadened
    
    def rotational_kernel(self, vsini, refwvl, dwl):
        '''Generate the rotational broadening kernel using the Gray profile (vectorized)
        
        Parameters
        ----------
        vsini : float
            Projected rotational velocity in km/s
        refwvl : float
            Reference wavelength [A].
        dwl : float
            The wavelength bin size [A].
        
        Returns
        -------
        kernel : array
            Convolution kernel for rotational broadening
        '''
        # Calculate delta wavelength and scaling factor
        self.vc = vsini / self.c
        dl = np.linspace(-self.vc * refwvl, self.vc * refwvl, int(2 * self.vc * refwvl / dwl) + 1)
        
        # Calculate the broadening profile
        self.dlmax = self.vc * refwvl
        self.c1 = 2. * (1. - self.eps) / (np.pi * self.dlmax * (1. - self.eps / 3.))
        self.c2 = self.eps / (2. * self.dlmax * (1. - self.eps / 3.))
        
        x = dl / self.dlmax
        kernel = np.zeros_like(dl)
        within_bounds = np.abs(x) < 1.0
        
        kernel[within_bounds] = (self.c1 * np.sqrt(1. - x[within_bounds]**2) +
                                self.c2 * (1. - x[within_bounds]**2))
        
        # Normalize the kernel to account for numerical accuracy
        kernel /= (np.sum(kernel) * dwl)
        return kernel


