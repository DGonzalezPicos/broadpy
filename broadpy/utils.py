import numpy as np
import pathlib
from astropy.io import fits

# get path of current script
path = pathlib.Path('/'.join(str(pathlib.Path(__file__).parent.absolute()).split('/')[:-1]))

def load_example_data(wave_range=(2320.0, 2330.0), jwst=False):
    
    # directory of script
    # path = '/'.join(str(pathlib.Path(__file__).parent.absolute()).split('/')[:-1])
    if jwst:
        wave_um, flux = np.load(str(path)+'/examples/data/model_teff1000K_logg4_R1e5.npy')
        wave = wave_um * 1e3 # [um] -> [nm]
        
    else:
        wave, flux = np.loadtxt(str(path)+'/examples/data/models_Teff_4300_logg_4.0_Z_+0.0.txt').T # T=4300K, logg=4.0, solar metallicity
    # select wavelength region in nm
    mask = (wave >= wave_range[0]) & (wave <= wave_range[1])
    assert np.any(mask), 'No data in the selected wavelength range'
    wave = wave[mask]
    flux = flux[mask]
    return wave, flux

def load_nirspec_resolution_profile(file=None, grating=None, wave=None):
    '''wave must be in [nm]'''
    
    assert (file is not None) or (grating is not None), 'Please provide a file or a grating'
    assert (file is None) or (grating is None), 'Please provide only a file or a grating'
    # print(f' Loading NIRSpec resolution profile for grating {grating} with file: {file}')
    if file is None:
        # file = path / f'../data/jwst_nirspec_{grating.lower()}_disp.fits'
        file = path / f'data/jwst_nirspec_{grating.lower()}_disp.fits'
    else:
        file = pathlib.Path(file)
        
    # check it's a fits file
    assert file.name.endswith('.fits'), 'Please provide a fits file'
    # check it exists
    assert file.exists(), f'File does not exist {file}'
    
    with fits.open(file) as hdul:
        # print(hdul.info())
        data = hdul[1].data
        wave_grid = data['WAVELENGTH']*1e3 # [um] -> [nm]
        # disp = data['DLDS'] # ignore for now
        resolution = data['R']
        
    if wave is None:
        return wave_grid, resolution
    else:
        return wave, np.interp(wave, wave_grid, resolution)