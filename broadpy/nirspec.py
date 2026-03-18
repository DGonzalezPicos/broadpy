from broadpy.instrument import InstrumentalBroadening
from broadpy.utils import load_nirspec_resolution_profile

import numpy as np
import pathlib
from typing import Dict, Iterable, Optional, Sequence, Tuple

def nirspec_broadening(
    wavelength: np.ndarray,
    flux: np.ndarray,
    grating: str = "g140h",
) -> np.ndarray:
    """Convenience wrapper around :class:`NIRSPec`."""
    return NIRSPec(wavelength=wavelength, flux=flux)(grating=grating)

class NIRSPec(InstrumentalBroadening):
    
    available_gratings = ["g140h", "g235h", "g395h"]
    wavelength_range_gratings = {
        "g140h": (970, 1820),
        "g235h": (1660, 3050),
        "g395h": (2870, 5280),
    }
    # numpy.polyfit output (highest power first) for resolution R(wavelength_nm).
    # Source: polynomial fit to bundled NIRSpec resolution profile FITS files in
    # this repository (see `data/jwst_nirspec_*_disp.fits`).
    _poly_coefficients_deg3 = {
        "g140h": [
            1.058543594223008702e-07,
            -1.994808808417648240e-04,
            2.027772970948571007e00,
            -3.763413376267637034e01,
        ],
        "g235h": [
            3.406828846855763279e-08,
            -1.540172668932293047e-04,
            1.400139327843935577e00,
            -1.802143551960200796e02,
        ],
        "g395h": [
            5.026866319623666920e-09,
            -3.076489882388708285e-05,
            7.470170985115294204e-01,
            -7.897977051813326455e01,
        ],
    }
    poly_degree_default = 3
    speed_of_light_km_s = 2.99792458e5  # km/s (used for c/R)
    # Set to True if you prefer extra validation on every call (costs an O(N) pass).
    validate_resolution_fit: bool = False

    def _get_poly_coefficients_path(self, poly_degree: int) -> pathlib.Path:
        """
        Return the path to the persisted polynomial coefficients file.

        Notes
        -----
        The coefficients file lives next to the NIRSpec FITS resolution profiles in
        the repository-level `data/` folder.
        """
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        return repo_root / "data" / f"nirspec_poly_coefficients_deg{poly_degree}.txt"

    def __init__(
        self,
        wavelength=None,
        flux=None,
        gratings: Optional[Sequence[str]] = ("g140h",),
    ):
        super().__init__(wavelength=wavelength, flux=flux)

        selected_gratings = list(np.atleast_1d(gratings))
        if selected_gratings:
            self.load_gratings(selected_gratings)

        # Cache computed FWHM arrays keyed by wavelength array identity.
        # This accelerates repeated broadening for the same wavelength grid.
        self._fwhm_cache_wavelength_id: Optional[int] = None
        self._fwhm_cache_by_grating: Dict[str, np.ndarray] = {}

    def load_gratings(self, gratings: Iterable[str]):
        invalid = [g for g in gratings if g not in self.available_gratings]
        if invalid:
            raise ValueError(
                f"Please provide valid gratings from {self.available_gratings}. Invalid: {invalid}"
            )

        # Keep track of all loaded gratings; do not drop previously loaded entries.
        existing_gratings = list(getattr(self, "gratings", []))
        combined_gratings = sorted(set(existing_gratings).union(set(gratings)))
        self.gratings = combined_gratings

        # Fast path: use embedded coefficients for the default polynomial degree.
        if self.poly_degree_default == 3:
            poly_coefficients = dict(getattr(self, "poly_coefficients", {}))
            for grating in self.gratings:
                poly_coefficients[grating] = np.asarray(
                    self._poly_coefficients_deg3[grating], dtype=float
                )
            self.poly_coefficients = poly_coefficients
            return

        # General path: load coefficients from disk (compute if missing).
        coefficients_path = self._get_poly_coefficients_path(
            poly_degree=self.poly_degree_default
        )
        if not coefficients_path.exists():
            compute_poly_coefficients(
                output_path=coefficients_path,
                poly_degree=self.poly_degree_default,
                gratings=self.available_gratings,
            )

        coefficients_by_grating, poly_degree = read_poly_coefficients(coefficients_path)
        if poly_degree != self.poly_degree_default:
            raise ValueError(
                "Polynomial degree mismatch between requested and stored coefficients. "
                f"Expected {self.poly_degree_default}, got {poly_degree}."
            )

        missing = [g for g in self.gratings if g not in coefficients_by_grating]
        if missing:
            compute_poly_coefficients(
                output_path=coefficients_path,
                poly_degree=self.poly_degree_default,
                gratings=self.available_gratings,
            )
            coefficients_by_grating, _ = read_poly_coefficients(coefficients_path)

        poly_coefficients = dict(getattr(self, "poly_coefficients", {}))
        for grating in self.gratings:
            poly_coefficients[grating] = coefficients_by_grating[grating]
        self.poly_coefficients = poly_coefficients

    def _ensure_grating_coefficients(self, grating: str) -> None:
        if grating not in self.available_gratings:
            raise ValueError(
                f"Please provide valid gratings from {self.available_gratings}. Invalid: {[grating]}"
            )
        if grating not in getattr(self, "poly_coefficients", {}):
            self.load_gratings([grating])
    
    def update_data(self, wavelength=None, flux=None):
        if wavelength is not None:
            self.wavelength = wavelength
            self.spacing = np.mean(
                2
                * np.diff(self.wavelength)
                / (self.wavelength[1:] + self.wavelength[:-1])
            )
            # Wavelength grid changed -> invalidate cached FWHM arrays.
            self._fwhm_cache_wavelength_id = id(self.wavelength)
            self._fwhm_cache_by_grating = {}
        if flux is not None:
            self.flux = flux
            
    
    def __call__(self, grating="g140h", wavelength=None, flux=None):
        self._ensure_grating_coefficients(grating)

        self.update_data(wavelength=wavelength, flux=flux)

        if not hasattr(self, "wavelength") or self.wavelength is None:
            raise ValueError("Please provide `wavelength`.")
        if not hasattr(self, "flux"):
            raise ValueError("Please provide `flux`.")
        if len(self.flux) != len(self.wavelength):
            raise ValueError("`wavelength` and `flux` must have the same length.")

        wavelength_to_use = self.wavelength

        # Compute (or reuse) the wavelength-dependent FWHM array.
        wavelength_id = id(wavelength_to_use)
        if self._fwhm_cache_wavelength_id == wavelength_id and grating in self._fwhm_cache_by_grating:
            fwhms_g = self._fwhm_cache_by_grating[grating]
        else:
            resolution_fit = np.polyval(
                self.poly_coefficients[grating], wavelength_to_use
            )
            if self.validate_resolution_fit:
                if np.any(resolution_fit <= 0):
                    raise ValueError(
                        "Polynomial resolution fit produced non-positive resolution values. "
                        "Check polynomial degree or provide wavelengths within the supported range."
                    )

            # FWHM = c / R, where R is lambda / delta_lambda.
            fwhms_g = self.speed_of_light_km_s / resolution_fit
            self._fwhm_cache_wavelength_id = wavelength_id
            self._fwhm_cache_by_grating[grating] = fwhms_g
        return super().__call__(fwhm=fwhms_g, kernel="gaussian_variable")
    
    def get_resolution_curve(self, grating="g140h", wavelength=None):
        self._ensure_grating_coefficients(grating)
        if wavelength is None:
            wavelength = self.wavelength
        if wavelength is None:
            raise ValueError("Please provide `wavelength`.")
        return np.polyval(self.poly_coefficients[grating], wavelength)


def compute_poly_coefficients(
    output_path: pathlib.Path,
    poly_degree: int,
    gratings: Sequence[str],
) -> pathlib.Path:
    """
    Compute polynomial coefficients describing the NIRSpec resolution profile.

    The polynomial fits resolution R(wavelength_nm) using `numpy.polyfit`:
    resolution_fit(wavelength_nm) = polyval(coefficients, wavelength_nm).

    Parameters
    ----------
    output_path:
        Plain-text file to write coefficients into.
    poly_degree:
        Degree of the polynomial.
    gratings:
        Grating identifiers to compute coefficients for.

    Returns
    -------
    pathlib.Path
        The written coefficients file path.
    """
    if poly_degree < 0:
        raise ValueError("`poly_degree` must be non-negative.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    coeffs_matrix_by_order = []
    for grating in gratings:
        wavelength_grid_nm, resolution_r = load_nirspec_resolution_profile(
            grating=grating
        )
        coeffs = np.polyfit(wavelength_grid_nm, resolution_r, poly_degree)
        coeffs_matrix_by_order.append(coeffs)

    # Shape: (poly_degree + 1, n_gratings); each column corresponds to a grating.
    coeffs_matrix = np.stack(coeffs_matrix_by_order, axis=1)

    header_lines = [
        "# nirspec_poly_coefficients",
        f"# poly_degree={poly_degree}",
        "# columns: " + " ".join(gratings),
        "# coefficients order: np.polyfit output (highest power first)",
    ]
    header = "\n".join(header_lines)

    with output_path.open("w", encoding="utf-8") as fp:
        fp.write(header + "\n")
        np.savetxt(fp, coeffs_matrix, fmt="%.18e")

    return output_path


def read_poly_coefficients(
    coefficients_path: pathlib.Path,
) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Read polynomial coefficients from a plain-text file.

    Returns
    -------
    tuple[dict[str, np.ndarray], int]
        (coefficients_by_grating, poly_degree)
    """
    if not coefficients_path.exists():
        raise FileNotFoundError(str(coefficients_path))

    with coefficients_path.open("r", encoding="utf-8") as fp:
        header_lines = []
        while True:
            pos = fp.tell()
            line = fp.readline()
            if not line.startswith("#"):
                fp.seek(pos)
                break
            header_lines.append(line.strip())

        degree: Optional[int] = None
        columns: Optional[Sequence[str]] = None
        for line in header_lines:
            if line.startswith("# poly_degree="):
                degree = int(line.split("=", 1)[1])
            if line.startswith("# columns:"):
                columns_part = line.split(":", 1)[1].strip()
                columns = [c for c in columns_part.split() if c]

        if degree is None:
            raise ValueError("Could not parse `poly_degree` from coefficients header.")

        if columns is None:
            raise ValueError(
                "Could not parse grating column names from coefficients header."
            )

        # Load numeric data (ignoring header comments).
        coeffs_matrix = np.loadtxt(fp)

    if coeffs_matrix.ndim == 1:
        coeffs_matrix = coeffs_matrix[:, None]

    expected_rows = degree + 1
    if coeffs_matrix.shape[0] != expected_rows:
        raise ValueError(
            "Unexpected coefficient matrix shape. "
            f"Expected {expected_rows} rows, got {coeffs_matrix.shape[0]}."
        )

    if coeffs_matrix.shape[1] != len(columns):
        raise ValueError(
            "Unexpected coefficient matrix shape. "
            f"Expected {len(columns)} columns, got {coeffs_matrix.shape[1]}."
        )

    coefficients_by_grating = {
        grating: coeffs_matrix[:, i] for i, grating in enumerate(columns)
    }
    return coefficients_by_grating, degree
    
    
    
    
            
if __name__ == '__main__':
    # test with random data
    import matplotlib.pyplot as plt
    from broadpy.utils import load_example_data
    wavelength, flux = load_example_data(wave_range=(940, 1940), jwst=True)

    # Compute (once) the polynomial coefficients used by NIRSPec at runtime.
    # For the default `poly_degree_default == 3` this uses embedded coefficients.
    n = NIRSPec(wavelength=wavelength, flux=flux, gratings=NIRSPec.available_gratings)
    
    fig, (ax_res, ax_residuals) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    
    for grating in NIRSPec.available_gratings:
        wave_grid_nm, resolution = load_nirspec_resolution_profile(grating=grating)
        resolution_fit = np.polyval(n.poly_coefficients[grating], wave_grid_nm)

        ax_res.plot(wave_grid_nm, resolution, label=f"{grating} (FITS)")
        ax_res.plot(
            wave_grid_nm,
            resolution_fit,
            label=f"{grating} (poly deg={n.poly_degree_default})",
            linestyle="--",
        )

        residuals = resolution - resolution_fit
        ax_residuals.plot(wave_grid_nm, residuals, label=f"{grating}")

        rms = np.sqrt(np.mean(residuals**2))
        print(f"[{grating}] RMS(R - R_fit) = {rms:.3e}")

    ax_res.set_ylabel("Resolution R = lambda/delta_lambda")
    ax_residuals.set_ylabel("Residuals")
    ax_residuals.axhline(0.0, color="k", linestyle="-")
    ax_res.legend()
    plt.show()
    