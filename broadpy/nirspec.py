from broadpy.instrument import InstrumentalBroadening
from broadpy.utils import load_nirspec_resolution_profile

class NIRSPec(InstrumentalBroadening):
    
    available_gratings = ['g140h', 'g235h', 'g395h']
    def __init__(self, x=None, y=None, gratings=['g140h']):
        super(NIRSPec, self).__init__(x, y)
        
        gratings = list(np.atleast_1d(gratings))
        if len(gratings) > 0:
            self.load_gratings(gratings)
        
        
    def load_gratings(self, gratings=['g140h','g235h', 'g395h']):
        
        assert all([grating in self.available_gratings for grating in gratings]), f'Please provide a valid grating: {self.available_gratings}'
        self.gratings = gratings

        # load resolution profiles and convert to FWHM [km/s]
        # self.fwhms = {g: 2.99792458e5 / load_nirspec_resolution_profile(grating=g)[1] for g in self.gratings}
        self.wave_grid = {}
        self.fwhms = {}
        for g in self.gratings:
            wg, fwhm_g = load_nirspec_resolution_profile(grating=g)
            self.wave_grid[g] = wg
            self.fwhms[g] = 2.99792458e5 / fwhm_g
        return self
    
    def update_data(self, x=None, y=None):
        if x is not None:
            self.x = x
            self.spacing = np.mean(2*np.diff(self.x) / (self.x[1:] + self.x[:-1]))
        if y is not None:
            self.y = y
            
    
    def __call__(self, grating='g140h', x=None, y=None):
        assert grating in self.fwhms.keys(), f'Please provide a valid grating: {self.gratings}'
        
        self.update_data(x=x, y=y)
                    
        x = x if x is not None else self.x
        assert x is not None, 'Please provide x'
        fwhms_g = np.interp(x, self.wave_grid[grating], self.fwhms[grating])
        return super(NIRSPec, self).__call__(fwhm=fwhms_g, kernel='gaussian_variable')
    
    
            
if __name__ == '__main__':
    # test with random data
    import numpy as np
    
    from broadpy.utils import load_example_data
    x, y = load_example_data(wave_range=(940, 1940), jwst=True)
    
    n = NIRSPec(gratings=['g140h'])
    y_broadened = n(grating='g140h', x=x, y=y)
    
    import matplotlib.pyplot as plt
    plt.plot(x, y)
    plt.plot(x, y_broadened)
    plt.show()
    