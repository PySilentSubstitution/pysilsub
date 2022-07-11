#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pysilsub.device
===============

A generic object oriented interface for multiprimary light stimulators.

@author: jtm
"""

import os
import os.path as op
from typing import List, Union, Optional, Tuple
import json
from pprint import pprint

import importlib_resources
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, basinhopping, OptimizeResult
from scipy.stats import beta
import matplotlib.pyplot as plt
import matplotlib.path as mplpath
from matplotlib.colors import cnames
import seaborn as sns
import pandas as pd
import numpy as np
from colour.plotting import plot_chromaticity_diagram_CIE1931

from pysilsub.CIE import (
    get_CIES026,
    get_CIE_1924_photopic_vl,
    get_CIE170_2_chromaticity_coordinates,
)
from pysilsub import colorfunc
from pysilsub.plotting import ss_solution_plot
from pysilsub.exceptions import StimulationDeviceError


class StimulationDevice:
    """Generic class for multiprimary stimultion device."""

    # Class attribute colors for photoreceptors
    photoreceptors = ["S", "M", "L", "R", "I"]

    aopic_colors = {
        "S": "tab:blue",
        "M": "tab:green",
        "L": "tab:red",
        "R": "tab:grey",
        "I": "tab:cyan",
    }

    # empty dict for curve fit params
    curveparams = {}

    def __init__(
        self,
        resolutions: List[int],
        colors: List[str],
        calibration: pd.DataFrame,
        wavelengths: List[int],
        action_spectra="CIES026",
        name: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> None:
        """Instantiate class for multiprimary light stimulation devices.

        Parameters
        ----------
        resolutions : list of int
            Resolution depth of primaries, i.e., the number of steps available
            for specifying the intensity of each primary. This is a list of
            integers to allow for systems where primaries may have different
            resolution depths. The number of elements in the list must
            equal the number of device primaries.
        colors : list of str
            List of valid color names for the primaries. Must be in
            `matplotlib.colors.cnames <https://matplotlib.org/stable/gallery/color/named_colors.html>`_.
        calibration : pd.DataFrame
            Spectral measurements to characterise the output of the device.
            Column headers must be wavelengths and each row a spectrum.
            Additional columns are needed to identify the Primary/Setting. For
            example, 380, ..., 780, Primary, Setting. See
            `here <https://pysilentsubstitution.github.io/pysilsub/03a_stimulation_devices.html#pysilsub.device.StimulationDevice>`_
            for further info.
        wavelengths : list, optional
            [start, stop, step] of wavelength range, e.g., [380, 781, 1] for
            380 to 780 in 1 nm bins. The default is None, in which case the
            wavelength range is determined from the calibration spds.
        action_spectra : str or pd.DataFrame
            Photoreceptor spectral sensetivities of the observer. By default,
            uses the CIES026 standard. Alternatively, pass a dataframe
            with user-defined values. See `CIE.get_CIES026()` and
            `pysilsub.asano` for inspiration.
        name : str
            Name of the device or stimulation system, e.g., `8-bit BCGAR
            photostimulator`. Defaults to `Stimulation Device`.

        Returns
        -------
        None

        """
        self.resolutions = resolutions
        self.colors = colors
        # self._check_color_names_valid()
        self.calibration = calibration
        self.wavelengths = wavelengths
        self._check_wavelengths()
        if action_spectra == "CIES026":
            self.action_spectra = get_CIES026(self.wavelengths[-1], fillna=True)
        else:
            self.action_spectra = action_spectra  # TODO: check valid
        self.name = name
        if self.name is None:
            self.name = "Stimulation Device"
        self.config = config

        # Create important data
        self.primaries = (
            self.calibration.index.get_level_values(0).unique().tolist()
        )
        self.nprimaries = len(self.resolutions)
        self.wls = self.calibration.columns

    def __str__(self):
        return f"""
    StimulationDevice:
        Resolutions: {self.resolutions},
        Colors: {self.colors},
        Calibration: {self.calibration.shape},
        Wavelengths: {self.wavelengths},
        Action Spectra: {self.action_spectra.shape},
        Name: {self.name}
    """

    @classmethod
    def from_json(cls, config_json):
        """Constructor that works from preformatted json config file.

        See `pysilsub.config` for further details.

        Parameters
        ----------

        config_json : str
            json file with configuration parameters for StimulationDevice.

        Returns
        -------
        pysilsub.StimulationDevice

        """
        config = json.load(open(config_json, "r"))

        # Get the calibration data
        calibration = cls.load_calibration_file(config["calibration_fpath"])

        return cls(
            resolutions=config["resolutions"],
            colors=config["colors"],
            calibration=calibration,
            wavelengths=config["wavelengths"],
            name=config["name"],
            config=config,
        )

    @classmethod
    def from_package_data(cls, example):
        """Constructor that uses example data from the package distribution.

        See ``show_package_data(verbose=True)`` for full details.

        Parameters
        ----------

        example : str
            Currently the following options are available:

                * 'STLAB_York'
                * 'STLAB_Oxford'
                * 'BCGAR'
                * 'VirtualSky'

        Example
        -------

        ``device = StimulationDevice.from_package_example('BCGAR')``

        Returns
        -------
        pysilsub.StimulationDevice

        """
        pkg = importlib_resources.files("pysilsub")
        json_config = example + ".json"
        config = json.load(open(pkg / "data" / json_config, "r"))
        dir_path = op.dirname(op.realpath(__file__))
        data_path = op.join(dir_path, "data", example + ".csv")
        calibration = cls.load_calibration_file(data_path)

        return cls(
            resolutions=config["resolutions"],
            colors=config["colors"],
            calibration=calibration,
            wavelengths=config["wavelengths"],
            name=config["name"],
            config=config,
        )

    @classmethod
    def show_package_data(cls, verbose=False):
        """Show details of example data in the package distribution.

        Parameters
        ----------
        verbose : bool
            Print detailed information if True. The default is False.

        Returns
        -------
        None.

        """
        pkg = importlib_resources.files("pysilsub")
        available = [f for f in os.listdir(pkg / "data") if f.endswith(".json")]

        for f in available:
            if not verbose:
                print(f.strip(".json"))
            else:
                config = json.load(open(pkg / "data" / f, "r"))
                print(f'{"*"*60}\n{config["name"]:*^60s}\n{"*"*60}')
                pprint(config)
                print("\n")

    @classmethod
    def load_calibration_file(cls, calibration_fpath):
        calibration = pd.read_csv(
            calibration_fpath, index_col=["Primary", "Setting"]
        )
        calibration.columns = calibration.columns.astype("int64")
        calibration.columns.name = "Wavelength"
        return calibration

    # Utils

    def _melt_spds(self, spds) -> pd.DataFrame:
        return spds.reset_index().melt(
            id_vars=["Primary", "Setting"],
            value_name="Flux",
            var_name="Wavelength (nm)",
        )

    # Error checking
    def _primary_input_is_valid(self, primary, primary_input):
        """Make sure primary input is sensible."""
        if primary not in self.primaries:
            raise StimulationDeviceError(
                f"Requested primary {primary} is unavailable for "
                f"{self.name}. The following primaries are available: "
                f"{self.primaries}"
            )

        if not isinstance(primary_input, (int, float)):
            raise TypeError("Must specify float or int for primary input.")

        msg = (
            f"Requested input {primary_input} exceeds "
            f"resolution of device primary {primary}"
        )

        if isinstance(primary_input, float):
            if not (primary_input >= 0.0 and primary_input <= 1.0):
                raise StimulationDeviceError(msg)

        elif isinstance(primary_input, int):
            if not (
                primary_input >= 0
                and primary_input <= self.resolutions[primary]
            ):
                raise StimulationDeviceError(msg)

        return True

    def _device_input_is_valid(self, device_input):
        """Make sure device input is sensible."""
        if len(device_input) != self.nprimaries:
            raise StimulationDeviceError(
                "Number of settings does not match number of device primaries."
            )
        if not (
            all(isinstance(s, int) for s in device_input)
            or all(isinstance(s, float) for s in device_input)
        ):
            raise StimulationDeviceError(
                "Can not mix float and int for device input."
            )

        if all(
            self._primary_input_is_valid(primary, primary_input)
            for primary, primary_input in enumerate(device_input)
        ):
            return True

    def _check_wavelengths(self):
        msg = "Wavelengths do not match spds"
        if not all(
            [wl in self.calibration.columns for wl in range(*self.wavelengths)]
        ):
            raise ValueError(msg)

    def _check_color_names_valid(self):
        request = f"Please choose from the following:\n {list(cnames.keys())}"
        msg = f"At least one color name was not valid. {request}"
        if not all([c in cnames.keys() for c in self.colors]):
            raise ValueError(msg)

    def _get_gamut(self):
        max_spds = self.calibration.loc[(slice(None), self.resolutions), :]
        XYZ = max_spds.apply(
            colorfunc.spd_to_XYZ, args=(self.wavelengths[-1],), axis=1
        )
        xy = (
            XYZ[["X", "Y"]]
            .div(XYZ.sum(axis=1), axis=0)
            .rename(columns={"X": "x", "Y": "y"})
        )
        xy = xy.append(xy.iloc[0], ignore_index=True)  # Join the dots
        return xy

    def _xy_in_gamut(self, xy_coord: Tuple[float]):
        """Return True if xy_coord is within the gamut of the device"""
        poly_path = mplpath.Path(self._get_gamut().to_numpy())
        return poly_path.contains_point(xy_coord)

    # Plotting functions
    def plot_action_spectra(
        self, ax: plt.Axes = None, **plt_kwargs
    ) -> plt.Axes:
        """Plot photoreceptor action spectra.

        Parameters
        ----------
        ax : plt.Axes, optional
            Axes on which to plot. The default is None.
        **plt_kwargs
            Options to pass to matplotlib plotting method..

        Returns
        -------
        ax : matplotlib.axes.Axes

        """
        if ax is None:
            ax = plt.gca()
        self.action_spectra.plot(color=self.aopic_colors, ax=ax, **plt_kwargs)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Spectral sensetivity")
        return ax

    def plot_gamut(
        self,
        ax: plt.Axes = None,
        show_1931_horseshoe: bool = True,
        show_CIE170_2_horseshoe: bool = True,
        **plt_kwargs,
    ) -> Union[plt.Figure, None]:
        """Plot the gamut of the stimulation device.

        Parameters
        ----------
        ax : plt.Axes, optional
            Axes on which to plot. The default is None.
        show_1931_horseshoe : bool, optional
            Whether to show the CIE1931 chromaticity horseshoe. The default is
            True.
        show_CIE170_2_horseshoe : bool, optional
            Whether to show the CIE170_2 chromaticity horseshoe. The default is
            True.

        Returns
        -------
        ax : matplotlib.axes.Axes

        """
        gamut = self._get_gamut()
        if ax is None:
            ax = plt.gca()
        if show_1931_horseshoe:
            plot_chromaticity_diagram_CIE1931(
                axes=ax, title=False, standalone=False
            )
        if show_CIE170_2_horseshoe:
            cie170_2 = get_CIE170_2_chromaticity_coordinates(connect=True)
            ax.plot(
                cie170_2["x"], cie170_2["y"], c="k", ls=":", label="CIE 170-2"
            )
        ax.plot(
            gamut["x"],
            gamut["y"],
            color="k",
            lw=2,
            marker="x",
            markersize=8,
            label="Gamut",
            **plt_kwargs,
        )
        ax.set(xlim=(-0.15, 0.9), ylim=(-0.1, 1), title=f"{self.name}")
        ax.legend()
        return ax

    def plot_calibration_spds(
        self, ax: plt.Axes = None, norm: bool = False, **plt_kwargs
    ) -> plt.Axes:
        """Plot the spectral power distributions for the stimulation device.

        Returns
        -------
        ax : matplotlib.axes.Axes

        """
        if ax is None:
            ax = plt.gca()
        if norm == True:
            data = self.calibration.div(self.calibration.max(axis=1), axis=0)
            ylabel = "Power (normalised)"
        elif norm == False:
            data = self.calibration
            try:
                ylabel = self.config["calibration_units"]
            except:
                ylabel = "Power"

        data = self._melt_spds(spds=data)

        _ = sns.lineplot(
            x="Wavelength (nm)",
            y="Flux",
            data=data,
            hue="Primary",
            palette=self.colors,
            units="Setting",
            ax=ax,
            lw=0.1,
            estimator=None,
            **plt_kwargs,
        )
        ax.set_title(f"{self.name} SPDs")
        ax.set_ylabel(ylabel)
        return ax

    def plot_multiprimary_spd(
        self,
        settings: Union[List[int], List[float]],
        ax: plt.Axes = None,
        name: Union[int, str] = 0,
        show_primaries: Optional[bool] = False,
        **plt_kwargs,
    ) -> plt.Figure:
        """Plot predicted spd for list of device settings.

        Parameters
        ----------
        settings : list of int or list of float
            List of settings for the device primaries. Must be of length
            `self.nprimaries` and consist entirely of float (0.-1.) or int
            (0-max resolution).
        ax : plt.Axes, optional
            Axes on which to plot. The default is None.
        name : int or str, optional
            A name for the spectrum. The default is 0.
        show_primaries : bool, optional
            Whether to show the primary components of the spectrum. The default
            is False.
        **plt_kwargs
            Options to pass to matplotlib plotting method.

        Returns
        -------
        ax : matplotlib.axes.Axes

        """
        if ax is None:
            ax = plt.gca()
        spd = self.predict_multiprimary_spd(settings, name)
        ax = spd.plot(c="k", label="SPD", **plt_kwargs)

        if show_primaries:
            primaries = self.predict_multiprimary_spd(settings, nosum=True)
            primaries.plot(ax=ax, **{"color": self.colors, "legend": False})
            ax.legend(title="Primary")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Power")
        return ax

    # Predict ouput
    def calculate_aopic_irradiances(self) -> pd.DataFrame:
        """Calculate aopic irradiances from calibration spds.

        Using the CIE026 spectral sensetivities, calculate alpha-opic
        irradiances (S, M, L, R, I) for every spectrum in `self.calibration`.

        Returns
        -------
        pd.DataFrame
            Alphaopic irradiances.

        """
        return self.calibration.dot(self.action_spectra)

    def calculate_lux(self):
        """Using the CIE1924 photopic luminosity function, calculate lux for
        every spectrum in `self.calibration`.

        Returns
        -------
        pd.DataFrame
            Lux values.

        """
        vl = get_CIE_1924_photopic_vl(binwidth=self.spd_binwidth[-1])
        lux = self.calibration.dot(vl.values) * 683  # lux conversion factor
        lux.columns = ["lux"]
        return lux

    def predict_primary_spd(
        self,
        primary: int,
        setting: Union[int, float],
        name: Union[int, str] = 0,
    ) -> np.ndarray:
        """Predict output for a single device primary at a given setting.

        This is the basis for all predictions.

        Parameters
        ----------
        primary : int
            Device primary.
        setting : int or float
            Device primary setting. Must be int (0-max resolution) or float
            (0.-1.).
        name : int or str, optional
            A name for the spectrum. The default is 0.

        Raises
        ------
        ValueError
            If requested value of setting exceeds resolution.

        Returns
        -------
        np.array
            Predicted spd for primary / setting.

        """
        if self._primary_input_is_valid(primary, setting):
            if isinstance(setting, float):
                # NOTE: Required for smoothness assumption of SLSQP! i.e.,
                # don't use an int. 
                setting = setting * self.resolutions[primary]

            # TODO: fix wavelength thing
            f = interp1d(
                x=self.calibration.loc[primary].index.values,
                y=self.calibration.loc[primary],
                axis=0,
                fill_value="extrapolate",
            )
            return pd.Series(f(setting), name=name, index=self.wls)

    def predict_multiprimary_spd(
        self,
        settings: Union[List[int], List[float]],
        name: Union[int, str] = 0,
        nosum: Optional[bool] = False,
    ) -> pd.Series:
        """Predict spectral power distribution of device for given settings.

        Predict the SPD output of the stimulation device for a given list of
        primary settings. Assumes linear summation of primaries.

        Parameters
        ----------
        settings : list of int or list of float
            List of settings for the device primaries. Must be of length
            `self.nprimaries` and consist entirely of float (0.-1.) or int
            (0-max resolution).
        name : int or str, optional
            A name for the spectrum, e.g. 'Background'. The default is 0.
        nosum : bool, optional
            Whether to sum the primary components of the spectrum.

        Raises
        ------
        ValueError
            If the number of elements in `settings` is greater than the number
            of device primaries.

            If the elements in `settings` are not exclusively int or float.

        Returns
        -------
        pd.DataFrame if nosum else pd.Series
            Predicted spectra or spectrum for given device settings.

        """
        if self._device_input_is_valid(settings):
            if name is None:
                name = 0
            spd = []
            for primary, setting in enumerate(settings):
                spd.append(self.predict_primary_spd(primary, setting, primary))
            spd = pd.concat(spd, axis=1)
            if nosum:
                spd.columns.name = "Primary"
                return spd
            else:
                spd = spd.sum(axis=1)
                spd.name = name
                return spd

    def predict_multiprimary_aopic(
        self, settings: Union[List[int], List[float]], name: Union[int, str] = 0
    ) -> pd.Series:
        """Predict a-opic irradiances of device for given settings.

        Parameters
        ----------
        settings : list of int or list of float
            List of settings for the device primaries. Must be of length
            `self.nprimaries` and consist entirely of float (0.-1.) or int
            (0-max resolution).
        name : int or str, optional
            A name for the output, e.g. 'Background'. The default is 0.

        Returns
        -------
        aopic : pd.DataFrame
            Predicted a-opic irradiances for given device settings.

        """
        if self._device_input_is_valid(settings):
            spd = self.predict_multiprimary_spd(settings, name=name)
            return spd.dot(self.action_spectra)

    def find_settings_xyY(
        self,
        xy: Union[List[float], Tuple[float]],
        luminance: float,
        tolerance: Optional[float] = 1e-6,
        plot_solution: Optional[bool] = False,
        verbose: Optional[bool] = True,
    ) -> OptimizeResult:
        """Find device settings for a spectrum with requested xyY values.

        Parameters
        ----------
        xy : List[float]
            Requested chromaticity coordinates (xy).
        luminance : float
            Requested luminance.
        tolerance : float, optional
            Acceptable precision for result.
        plot_solution : bool, optional
            Set to True to plot the solution. The default is False.
        verbose : bool, optional
            Set to True to print status messages. The default is False.

        Returns
        -------
        result : OptimizeResult
            The result of the optimisation procedure, with result.x as the
            settings that will produce the spectrum.

        """
        if len(xy) != 2:
            raise ValueError("xy must be of length 2.")

        if not self._xy_in_gamut(xy):
            print("WARNING: specified xy coordinates are outside of")
            print("the device's gamut. Searching for closest match.")
            print("This could take a while, and results may be useless.\n")

        requested_xyY = colorfunc.xy_luminance_to_xyY(xy, luminance)
        requested_LMS = colorfunc.xyY_to_LMS(requested_xyY)

        # Objective function to find device settings for given xyY
        def _xyY_objective_function(x0: List[float]):
            aopic = self.predict_multiprimary_aopic(x0)
            return sum(
                pow(requested_LMS - aopic[["L", "M", "S"]].to_numpy(), 2)
            )

        # Bounds
        bounds = [(0.0, 1.0,) for primary in self.resolutions]
        # Arguments for local solver
        minimizer_kwargs = {
            "method": "SLSQP",
            "bounds": bounds,
            "options": {"maxiter": 500},
        }

        # Callback for global search
        def _callback(x, f, accepted):
            if accepted and tolerance is not None:
                if f < tolerance:
                    return True

        # Random starting point
        x0 = np.random.uniform(0, 1, self.nprimaries)

        # Do global search
        result = basinhopping(
            func=_xyY_objective_function,
            x0=x0,
            niter=100,
            T=1.0,
            stepsize=0.5,
            minimizer_kwargs=minimizer_kwargs,
            take_step=None,
            accept_test=None,
            callback=_callback,
            interval=50,
            disp=True,
            niter_success=None,
            seed=None,
        )

        # TODO: refactor this
        solution_lms = self.predict_multiprimary_aopic(result.x)[
            ["L", "M", "S"]
        ].values
        solution_xyY = colorfunc.LMS_to_xyY(solution_lms)
        print(f"Requested LMS: {requested_LMS}")
        print(f"Solution LMS: {solution_lms}")

        # Reacfactor!
        if plot_solution is not None:
            fig, axs = ss_solution_plot()
            # Plot the spectrum
            self.predict_multiprimary_spd(
                result.x, name=f"solution_xyY:\n{solution_xyY.round(3)}"
            ).plot(ax=axs[0], legend=True, c="k")
            self.predict_multiprimary_spd(
                result.x,
                name=f"solution_xyY:\n{solution_xyY.round(3)}",
                nosum=True,
            ).plot(ax=axs[0], color=self.colors, legend=False)
            axs[1].scatter(
                x=requested_xyY[0],
                y=requested_xyY[1],
                s=100,
                marker="o",
                facecolors="none",
                edgecolors="k",
                label="Requested",
            )
            axs[1].scatter(
                x=solution_xyY[0],
                y=solution_xyY[1],
                s=100,
                c="k",
                marker="x",
                label="Resolved",
            )
            self.plot_gamut(ax=axs[1], show_CIE170_2_horseshoe=False)
            axs[1].legend()
            device_ao = self.predict_multiprimary_aopic(
                result.x, name="Background"
            )
            colors = [val[1] for val in self.aopic_colors.items()]
            device_ao.plot(kind="bar", color=colors, ax=axs[2])

        return result

    def do_gamma(self):
        """Make a gamma table from cumulative distribution functions.

        """

        # Curve fitting function
        def func(x, a, b):
            return beta.cdf(x, a, b)

        gamma_table = []
        for primary, df in self.calibration.groupby(level=0):
            xdata = (
                df.index.get_level_values(1) / self.resolutions[primary]
            ).to_series()
            ydata = df.sum(axis=1)
            ydata = ydata / ydata.max()

            popt, pcov = curve_fit(beta.cdf, xdata, ydata, p0=[2.0, 1.0])
            self.curveparams[primary] = popt
            ypred = func(xdata, *popt)

            # interp func for new ydata, inverting the values
            new_y_func = interp1d(ypred, xdata)

            # Interpolate for full lookup table
            resolution = self.resolutions[primary]
            new_y = new_y_func(np.linspace(0, 1, resolution + 1)) * resolution
            gamma_table.append(new_y.astype("int"))

        gamma = pd.DataFrame(gamma_table)

        self.gamma = gamma

    # TODO: decide whether to keep these

    def fit_curves(self):
        """Fit curves to the unweighted irradiance of spectral measurements
        and save the parameters.

        Returns
        -------
        fig
            Figure.

        """
        # breakpoint()
        # plot
        ncols = self.nprimaries
        fig, axs = plt.subplots(
            nrows=1, ncols=ncols, figsize=(16, 6), sharex=True, sharey=True
        )
        # axs = [item for sublist in axs for item in sublist]

        for primary in range(self.nprimaries):
            xdata = (
                self.calibration.loc[primary].index / self.resolutions[primary]
            ).to_series()
            ydata = self.calibration.loc[primary].sum(axis=1)
            ydata = ydata / np.max(ydata)

            # Curve fitting function
            def func(x, a, b):
                return beta.cdf(x, a, b)

            axs[primary].scatter(xdata, ydata, color=self.colors[primary], s=2)

            # Fit
            popt, pcov = curve_fit(beta.cdf, xdata, ydata, p0=[2.0, 1.0])
            self.curveparams[primary] = popt
            ypred = func(xdata, *popt)
            axs[primary].plot(
                xdata,
                ypred,
                color=self.colors[primary],
                label="fit: a=%5.3f, b=%5.3f" % tuple(popt),
            )
            axs[primary].set_title("Primary {}".format(primary))
            axs[primary].legend()

        for ax in axs:
            ax.set_ylabel("Output fraction (irradiance)")
            ax.set_xlabel("Input fraction")

        plt.tight_layout()

        return fig

    def optimise(
        self, primary: int, settings: Union[List[int], List[float]]
    ) -> Union[List[int], List[float]]:
        """Optimise a stimulus profile by applying the curve parameters.

        Parameters
        ----------
        primary : int
            Primary being optimised.
        settings : np.array
            Array of intensity values to optimise for specified LED.

        Returns
        -------
        np.array
            Optimised intensity values.

        """
        if not self.curveparams:
            print("No parameters yet. Run .fit_curves(...) first...")
        params = self.curveparams[primary]
        settings = self.settings_to_weights(settings)
        optisettings = beta.ppf(settings, params[0], params[1])
        return self.weights_to_settings(optisettings)

    def gamma_lookup(self, led, setting):
        return int(self.gamma.loc[led, setting])

    def gamma_correct(self, settings):
        return [
            self.gamma_lookup(led, setting)
            for led, setting in enumerate(settings)
        ]

    def settings_to_weights(self, settings: List[int]) -> List[float]:
        """Convert a list of settings to a list of weights.

        Parameters
        ----------
        settings : List[int]
            List of length ``self.nprimaries`` containing device settings 
            ranging from 0 to max-resolution for each primary.

        Returns
        -------
        list
            List of primary weights.

        """
        if self._device_input_is_valid(settings):
            if not all(isinstance(s, int) for s in settings):
                raise StimulationDeviceError(
                    "settings_to_weights expects int but got float."
                )
            return [float(s / r) for s, r in zip(settings, self.resolutions)]

    def weights_to_settings(self, weights: List[float]) -> List[int]:
        """Convert a list of channel input weights to a list of settings.

        Parameters
        ----------
        weights : List[float]
            List of length ``self.nprimaries`` containing input weights,
            ranging from 0. to 1., for each primary.

        Returns
        -------
        list
            List of primary settings.

        """
        if self._device_input_is_valid(weights):
            if not all(isinstance(w, float) for w in weights):
                raise StimulationDeviceError(
                    "weights_to_settings expects float but got int."
                )
            return [int(w * r) for w, r in zip(weights, self.resolutions)]
