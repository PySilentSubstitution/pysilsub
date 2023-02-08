#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

pysilsub.jazcal
===============

A convenience module for processing data with an OceanOptics JAZ spectrometer.

"""

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy import interpolate


class HL_2000_CAL:
    """Halogen light source used for obtaining measurements.

    Bought in 2011. Possibly exceeded 50 hours bulb time.

    Attributes
    ----------
    serial_number: str
        Serial number of the HL_2000_CAL light source.
    cc: NDArray
        Known output of light source at given wavelengths in uW/cm^2/nm when
        sampled with a cosine corrector.
    fib: NDArray
        Known output of light source at given wavelengths in uW/cm^2/nm when
        sampled with a fiber optic probe.

    """

    # Serial number of the light source
    serial_number: str = "030410313"

    # Calibration values with cosine corrector
    cc: NDArray = np.array(
        [
            [3.0000e02, 2.2874e-01],
            [3.1000e02, 1.6599e-01],
            [3.2000e02, 1.6086e-01],
            [3.3000e02, 1.6554e-01],
            [3.4000e02, 1.3879e-01],
            [3.5000e02, 2.1906e-01],
            [3.6000e02, 3.2334e-01],
            [3.7000e02, 4.1326e-01],
            [3.8000e02, 5.6613e-01],
            [3.9000e02, 7.5291e-01],
            [4.0000e02, 9.4661e-01],
            [4.2000e02, 1.5044e00],
            [4.4000e02, 2.2789e00],
            [4.6000e02, 3.3120e00],
            [4.8000e02, 4.5464e00],
            [5.0000e02, 6.0492e00],
            [5.2500e02, 8.3577e00],
            [5.5000e02, 1.1067e01],
            [5.7500e02, 1.4128e01],
            [6.0000e02, 1.7633e01],
            [6.5000e02, 2.5670e01],
            [7.0000e02, 3.4804e01],
            [7.5000e02, 4.8149e01],
            [8.0000e02, 6.3828e01],
            [8.5000e02, 8.0206e01],
            [9.0000e02, 9.8631e01],
            [9.5000e02, 1.1961e02],
            [1.0000e03, 1.4410e02],
            [1.0500e03, 1.6983e02],
        ]
    )

    # Calibration values with craw fiber
    fib: NDArray = np.array(
        [
            [3.0000e02, 1.0694e-03],
            [3.1000e02, 5.7952e-04],
            [3.2000e02, 8.3353e-04],
            [3.3000e02, 7.5975e-04],
            [3.4000e02, 1.4024e-03],
            [3.5000e02, 1.6770e-03],
            [3.6000e02, 2.1283e-03],
            [3.7000e02, 3.3981e-03],
            [3.8000e02, 4.1828e-03],
            [3.9000e02, 6.0289e-03],
            [4.0000e02, 8.1559e-03],
            [4.2000e02, 1.3690e-02],
            [4.4000e02, 2.2714e-02],
            [4.6000e02, 3.6709e-02],
            [4.8000e02, 5.6659e-02],
            [5.0000e02, 7.9038e-02],
            [5.2500e02, 1.1952e-01],
            [5.5000e02, 1.7005e-01],
            [5.7500e02, 2.2761e-01],
            [6.0000e02, 2.9340e-01],
            [6.5000e02, 4.5449e-01],
            [7.0000e02, 6.5138e-01],
            [7.5000e02, 9.5320e-01],
            [8.0000e02, 1.3334e00],
            [8.5000e02, 1.8282e00],
            [9.0000e02, 2.4976e00],
            [9.5000e02, 3.5639e00],
            [1.0000e03, 5.6178e00],
            [1.0500e03, 8.2277e00],
        ]
    )

    def __init__(self):
        pass

    def resample_calibration_data(self, wavelength_range, which="fib"):
        if which == "fib":
            f = interpolate.interp1d(self.fib[:, 0], self.fib[:, 1])
        elif which == "cc":
            f = interpolate.interp1d(self.cc[:, 0], self.cc[:, 1])

        return pd.Series(f(wavelength_range), index=wavelength_range)


class EasyJAZ:
    """Convenience class for JAZ spectrometer at the University of York."""

    serial_number = "JAZA1505"

    # Wavelength coefficients - calculated against spectral lines of argon lamp
    # These now exist on the spectrometer itself
    wavelength_coefficients = np.array(
        [
            338.5374924539017,
            0.3802367113739599,
            -1.5959354792685768e-05,
            -2.507246200675931e-09,
        ]
    )

    # For a dark measurement, the following pixels report over mean + 2*std
    # counts
    bad_pixels = np.array(
        [
            356.70624319,
            362.77315254,
            382.80601801,
            451.93989709,
            474.45564311,
            503.76619535,
            517.6001837,
            525.94525811,
            570.20602528,
            588.72787654,
            599.00727736,
            652.9790287,
            659.56465689,
            665.78855964,
            701.11649374,
            702.81887507,
            719.77774393,
            787.06092921,
            788.3728893,
            855.14726799,
            863.37597911,
            870.93966241,
            918.42288312,
            923.61911093,
            936.38738068,
            980.28497852,
            991.42894554,
            994.34745545,
        ]
    )

    def __init__(self, collection_area=0.4):
        self.collection_area = collection_area

    def get_bad_pixel_idxs(self):
        return [np.where(self.wls == pixel)[0][0] for pixel in self.bad_pixels]

    def get_nm_per_pixel(self):
        """Calculate width of each pixel in nanometers."""
        return np.hstack(
            [
                (self.cal_wls[1] - self.cal_wls[0]),
                (self.cal_wls[2:] - self.cal_wls[:-2]) / 2,
                (self.cal_wls[-1] - self.cal_wls[-2]),
            ]
        )

    # TODO - for a single measurement
    def process_measurement(self, spectrum, integration_time):
        pass

    def process_measurements(self, spectra, info, box_car_smoothing_window=1):
        """Process to visible wavelengths."""
        # Set bad pixels to nan and interpolate
        spectra.iloc[:, self.get_bad_pixel_idxs()] = np.nan
        spectra = spectra.interpolate(axis=1)

        # Replace with calibrated wavelengths
        spectra.columns = self.cal_wls

        # Counts per nanometer
        spectra = spectra.div(self.get_nm_per_pixel())

        # Divide by collection area
        spectra = spectra / self.collection_area

        # Counts per nanometer per second
        spectra = spectra.div((info.integration_time / (1000 * 1000)), axis=0)

        # Get rid of negative values
        spectra[spectra < 0] = 0

        # Resample to visible wavelengths
        f = interpolate.interp1d(
            spectra.columns, spectra.to_numpy(), fill_value="extrapolate"
        )
        visible_wavelengths = np.arange(380, 781, 1)
        visible_spds = f(visible_wavelengths)
        visible_spds = pd.DataFrame(visible_spds, columns=visible_wavelengths)

        return visible_spds


# if __name__ == "__main__":
#     jaz = EasyJAZ()
#     spectra = pd.read_csv(
#         "/home/bbsrc/Code/BakerWadeBBSRC/data/ProPixx/fo_gersun_halogenjaz_spectra.csv"
#     )
#     info = pd.read_csv(
#         "/home/bbsrc/Code/BakerWadeBBSRC/data/ProPixx/fo_gersun_halogenjaz_info.csv"
#     )

#     processed = jaz.process_measurements(spectra, info)
#     processed.index = ["FibreOptic", "Gershun"]
#     # processed.to_csv('/Users/jtm545/Projects/BakerWadeBBSRC/data/ProPixx/fo_gershun_halogen_jaz_processed.csv')
