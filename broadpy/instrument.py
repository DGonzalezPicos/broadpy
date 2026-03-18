import numpy as np
from scipy.ndimage import convolve1d # suitable for 'same' output shape as input
from scipy.special import voigt_profile

class InstrumentalBroadening:
    
    c = 2.998e5 # km/s
    sqrt8ln2 = np.sqrt(8 * np.log(2))
    
    available_kernels = ['gaussian',
                         'lorentzian', 
                         'voigt',
                         'gaussian_variable',
                         'auto']
    
    def __init__(self, wavelength=None, flux=None):
        if wavelength is not None:
            self.wavelength = wavelength  # units of wavelength
            self.spacing = np.mean(
                2 * np.diff(self.wavelength) / (self.wavelength[1:] + self.wavelength[:-1])
            )

        if flux is not None:
            if wavelength is None:
                raise ValueError("`wavelength` must be provided when setting `flux`.")
            self.flux = flux  # units of flux (does not matter)
            assert len(self.wavelength) == len(self.flux), "`wavelength` and `flux` should match."

    
    def __call__(self, resolution=None, fwhm=None, gamma=None, truncate=4.0, kernel='auto'):
        '''Instrumental broadening
        
        provide either instrumental resolutionolution lambda/delta_lambda or FWHM in km/s'''
        kernel = self.__read_kernel(resolution=resolution, fwhm=fwhm, gamma=gamma) if kernel == 'auto' else kernel
        assert kernel in self.available_kernels, f'Please provide a valid kernel: {self.available_kernels}'
        # print(f' Applying {kernel} kernel')
        
        if kernel in ['gaussian', 'voigt']:
            fwhm = fwhm if fwhm is not None else (self.c / resolution)

        if kernel == 'gaussian':
            _kernel = self.gaussian_kernel(fwhm, truncate)
        
        if kernel == 'lorentzian':
            assert gamma is not None, 'Please provide a gamma value for the Lorentzian kernel'
            _kernel = self.lorentz_kernel(gamma, truncate)
        
        if kernel == 'voigt':
            assert gamma is not None, 'Please provide a gamma value for the Lorentzian kernel'        
            _kernel = self.voigt_kernel(fwhm, gamma, truncate)
            
        if kernel == 'gaussian_variable':
            fwhm_array = np.asarray(fwhm, dtype=float)
            assert fwhm_array.ndim == 1, 'Please provide a 1D list/array of FWHM values'
            assert len(fwhm_array) == len(self.wavelength), (
                'FWHM list should have the same length as the wavelength array'
            )
            _kernels, lw = self.gaussian_variable_kernel(fwhm_array, truncate)

            # Robust edge + bad-pixel treatment:
            # - `edge` padding avoids mirrored spectral features at boundaries.
            # - weighted normalization ignores NaN pixels without biasing flux scale.
            flux_values = np.asarray(self.flux, dtype=float)
            valid_mask = np.isfinite(flux_values).astype(float)
            flux_filled = np.nan_to_num(flux_values, nan=0.0)

            flux_pad = np.pad(flux_filled, (lw, lw), mode='edge')
            valid_pad = np.pad(valid_mask, (lw, lw), mode='edge')

            flux_windows = np.lib.stride_tricks.sliding_window_view(
                flux_pad,
                window_shape=(2 * lw + 1),
            )
            valid_windows = np.lib.stride_tricks.sliding_window_view(
                valid_pad,
                window_shape=(2 * lw + 1),
            )

            weighted_flux = np.einsum('ij,ij->i', _kernels, flux_windows)
            weighted_norm = np.einsum('ij,ij->i', _kernels, valid_windows)

            y_lsf = np.divide(
                weighted_flux,
                weighted_norm,
                out=np.full_like(weighted_flux, np.nan),
                where=weighted_norm > 1e-12,
            )
            return y_lsf
            
        y_lsf = convolve1d(self.flux, _kernel, mode='nearest')
        return y_lsf
    
    @classmethod
    def gaussian_profile(self, x, x0, sigma):
        '''Gaussian function'''
        return np.exp(-0.5 * ((x - x0) / sigma)**2)# / (sigma * np.sqrt(2*np.pi))
    
    @classmethod
    def lorentz_profile(self, x, x0, gamma):
        '''Lorentzian function'''
        return gamma / np.pi / ((x - x0)**2 + gamma**2)

    
    def gaussian_kernel(self, 
                        fwhm,
                        truncate=4.0, 
                        ):
        ''' Gaussian kernel
        
        Parameters
        ----------
        fwhm : float
            Full width at half maximum of the Gaussian kernel in km/s
        truncate : float
            Truncate the kernel at this many standard deviations from the mean (default: 4.0)
        
        Returns
        -------
        kernel : array
            Convolution kernel
        '''
        # Adapted from scipy.ndimage.gaussian_filter1d        
        sd = (fwhm/self.c) / self.sqrt8ln2 / self.spacing
        lw = int(truncate * sd + 0.5)
    
        kernel_x = np.arange(-lw, lw+1)
        kernel = self.gaussian_profile(kernel_x, 0, sd)
        kernel /= np.sum(kernel)  # normalize the kernel
        return kernel
    
    def gaussian_variable_kernel(self, fwhm, truncate=4.0):
        ''' Gaussian kernel with variable FWHM
        
        Parameters
        ----------
        fwhm : array
            Full width at half maximum of the Gaussian kernel in km/s
        truncate : float
            Truncate the kernel at this many standard deviations from the mean (default: 4.0)
        
        Returns
        -------
        kernel : array
            Convolution kernel
        '''
        sd = (fwhm/self.c) / self.sqrt8ln2 / self.spacing
        if np.any(~np.isfinite(sd)):
            raise ValueError('FWHM values must be finite.')
        if np.any(sd <= 0):
            raise ValueError('FWHM values must be strictly positive.')

        # Lower bound prevents division-by-zero and over-sharp kernels.
        sd = np.maximum(sd, 1e-6)
        lw = int(truncate * sd.max() + 0.5)
        x = np.arange(-lw, lw + 1)
        
        # Use broadcasting to create a 2D array of Gaussian kernels
        kernels = np.exp(-0.5 * (x[None, :] / sd[:, None]) ** 2)
        kernel_norm = kernels.sum(axis=1)[:, None]
        kernels /= np.where(kernel_norm > 0, kernel_norm, 1.0)
        return kernels, lw
        
    
    def lorentz_kernel(self, gamma, truncate=4.0):
        ''' Lorentzian kernel
        Parameters
        ----------
        gamma : float
            Full width at half maximum of the Lorentzian kernel in km/s
        truncate : float
            Extent of the kernel as a multiple of gamma (default: 5)
        
        Returns
        -------
        kernel : array
            Convolution kernel
        ''' 
        gamma_pixels = gamma / self.c / self.spacing
        lw = int(truncate * gamma_pixels + 0.5)
        
        kernel_x = np.arange(-lw, lw+1)
        kernel = self.lorentz_profile(kernel_x, 0, gamma_pixels)
        kernel /= np.sum(kernel)
        return kernel
    
    
    def voigt_kernel(self, fwhm, gamma, truncate=4.0):
        ''' Voigt kernel using scipy.special.voigt_profile
        
        Parameters
        ----------
        fwhm : float
            Full width at half maximum of the Gaussian kernel in km/s
        gamma : float
            Half width at half maximum of the Lorentzian kernel in km/s
        truncate : float
            Extent of the kernel as a multiple of the standard deviation (default: 4.0)
        
        Returns
        -------
        kernel : array
            Convolution kernel
        '''        
        sigma = (fwhm / self.c) / self.sqrt8ln2 / self.spacing
        gamma_pixels = gamma / self.c / self.spacing
        lw = int(truncate * max(sigma, gamma_pixels) + 0.5)
        
        # Define the kernel range
        kernel_x = np.arange(-lw, lw + 1)

        # Create the Voigt profile using scipy.special.voigt_profile
        kernel = voigt_profile(kernel_x, sigma, gamma_pixels)
        
        # Normalize the kernel
        kernel /= np.sum(kernel)
        
        return kernel
    
    def __read_kernel(self, resolution=None, fwhm=None, gamma=None):
        '''Read kernel from the input parameters'''
        if resolution is not None:
            return 'gaussian'
        if fwhm is not None and isinstance(fwhm, (list, np.ndarray)):
            return 'gaussian_variable'
        if fwhm is not None and gamma is None:
            return 'gaussian'
        if fwhm is None and gamma is not None:
            return 'lorentzian'
        if fwhm is not None and gamma is not None:
            return 'voigt'
        raise ValueError(f'Please provide a valid kernel: {self.available_kernels}')
