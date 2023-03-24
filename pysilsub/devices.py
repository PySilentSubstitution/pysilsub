"""
``pysilsub.devices``
====================

Software model for multiprimary stimulation devices.

"""
from __future__ import annotations

import os
import os.path as op
from typing import Any, Sequence, Union, List, Tuple, Type
import json
import pprint

import importlib_resources
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import numpy.typing as npt
import colour

from . import colorfuncs
from . import CIE
from . import observers


# Type aliases
PrimaryInput = Union[int, float]
PrimaryWeights = Sequence[float]
PrimarySettings = Sequence[int]
DeviceInput = Union[PrimarySettings, PrimaryWeights]
PrimaryBounds = List[Tuple[float, float]]
PrimaryColors = Union[Sequence[str], Sequence[Sequence[Union[int, float]]]]


class StimulationDeviceError(Exception):
    """Generic Python-exception-derived object.

    Raised programmatically by ``StimulationDevice`` methods when arguments do
    not agree with internal parameters and those used to instantiate the class.

    """


class StimulationDevice:
    """Generic class for calibrated multiprimary stimultion device.

    Includes methods to visualise and predict data for a calibrated
    multiprimary system.

    """

    def __init__(
        self,
        calibration: str | pd.DataFrame,
        calibration_wavelengths: Sequence[int],
        primary_resolutions: Sequence[int],
        primary_colors: PrimaryColors,
        observer: str
        | Type[observers._Observer]
        | None = "CIE_standard_observer",
        name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Instantiate class with specified parameters.

        Parameters
        ----------
        calibration : str or pd.DataFrame
            DataFrame or path to CSV file with spectral measurements to
            characterise the output of the device. Column headers must be
            wavelengths and each row a spectrum. Additional columns are needed
            to identify the Primary/Setting. For example,
            380, 381, ..., 780, Primary, Setting. See
            `here <https://pysilentsubstitution.github.io/pysilsub/03a_stimulation_devices.html#pysilsub.device.StimulationDevice>`_
            for further info. See also `cls.load_calibration_file()`
        calibration_wavelengths : sequence of int
            [start, stop, step] of wavelength range, e.g., [380, 781, 1] for
            380 to 780 in 1 nm bins. The default is None, in which case the
            wavelength range is determined from the calibration spds.
        primary_resolutions : sequence of int
            Resolution depth of primaries, i.e., the number of steps available
            for specifying the intensity of each primary. This is a list of
            integers to allow for systems where primaries may have different
            resolution depths. The number of elements must equal the number of
            device primaries.
        primary_colors : sequence of str or rgb
            Sequence of valid color names for the primaries. Must be in
            `matplotlib.colors.cnames <https://matplotlib.org/stable/gallery/color/named_colors.html>`_.
            Alternatively, pass a sequence of RGB values.
        observer : str or pysilsub.observer.Observer
            Colorimetric observer model. The default is "CIE_standard_observer"
            , in which case defaults to the CIE standard colorimetric observer
            model (age=32, field_size=10), but can pass a custom observer
            model.
        name : str, optional
            Name of the device or stimulation system, e.g., `8-bit BCGAR
            photostimulator`. The default name is `Stimulation Device`.
        config : dict, optional
            Additional information from the config file. The default is
            ``None``.

        Returns
        -------
            None

        """
        if isinstance(calibration, str):
            self.calibration = self.load_calibration_file(calibration)
        else:
            self.calibration = calibration

        self.primaries = (
            self.calibration.index.get_level_values(0).unique().tolist()
        )
        self.nprimaries = len(self.primaries)

        self._assert_wavelengths_are_valid(calibration_wavelengths)
        self.calibration_wavelengths = calibration_wavelengths

        self._assert_resolutions_are_valid(primary_resolutions)
        self.primary_resolutions = primary_resolutions

        self._assert_colors_are_valid(primary_colors)
        self.primary_colors = primary_colors

        self._assert_observer_is_valid(observer)
        if observer == "CIE_standard_observer":
            self.observer = observers.ColorimetricObserver()
        else:
            self.observer = observer

        self.name = name
        if self.name is None:
            self.name = "Stimulation Device"

        self.config = config

        self.gamma = None

    def __str__(self):
        return f"""
    StimulationDevice:
        Calibration: {self.calibration.shape},
        Calibration Wavelengths: {self.calibration_wavelengths},
        Primary Resolutions: {self.primary_resolutions},
        Primary Colors: {self.primary_colors},
        Observer: {self.observer},
        Name: {self.name}
    """

    @classmethod
    def from_json(cls, config_json: str) -> StimulationDevice:
        """Construct class from preformatted json config file.

        See `pysilsub.config` for further details.

        Parameters
        ----------
        config_json : str
            json file with configuration parameters for `StimulationDevice`.

        Returns
        -------
        class
            `pysilsub.StimulationDevice`

        """
        config = json.load(open(config_json, "r"))

        calibration = cls.load_calibration_file(config["calibration"])

        return cls(
            calibration=calibration,
            calibration_wavelengths=config["calibration_wavelengths"],
            primary_resolutions=config["primary_resolutions"],
            primary_colors=config["primary_colors"],
            name=config["name"],
            config=config,
        )

    @classmethod
    def from_package_data(cls, example: str) -> StimulationDevice:
        """Construct class from example package data.

        For further details::

            show_package_data(verbose=True)

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
        class
            `pysilsub.StimulationDevice`

        """
        pkg = importlib_resources.files("pysilsub")
        json_config = example + ".json"
        config = json.load(open(pkg / "data" / json_config, "r"))
        dir_path = op.dirname(op.realpath(__file__))
        data_path = op.join(dir_path, "data", example + ".csv")
        calibration = cls.load_calibration_file(data_path)

        return cls(
            calibration=calibration,
            calibration_wavelengths=config["calibration_wavelengths"],
            primary_resolutions=config["primary_resolutions"],
            primary_colors=config["primary_colors"],
            name=config["name"],
            config=config,
        )

    # TODO: decide what this does
    @classmethod
    def show_package_data(cls, verbose: bool = False) -> list[str]:
        """Show details of example data in the package distribution.

        Parameters
        ----------
        verbose : bool, optional
            Print detailed information if True. The default is False.

        Returns
        -------
        list
            Available package data

        """
        pkg = importlib_resources.files("pysilsub")
        available = [
            f for f in os.listdir(pkg / "data") if f.endswith(".json")
        ]

        if verbose:
            for f in available:
                config = json.load(open(pkg / "data" / f, "r"))
                print(f'{"*"*60}\n{config["name"]:*^60s}\n{"*"*60}')
                pprint.pprint(config)
                print("\n")

        return [a.strip(".json") for a in available]

    @classmethod
    def load_calibration_file(cls, calibration_fpath: str) -> pd.DataFrame:
        """Load calibration file into pandas DataFrame.

        First row must contain column headers *Primary* and *Setting* and
        numbers to describe the wavelength sampling (e.g., Primary, Setting,
        380, 381, ..., 780). All remaining rows row should contain a spectral
        measurement and the corresponding values for *Primary* and *Setting*
        to identify the measurement.

        Parameters
        ----------
        calibration_fpath : str
            Path to CSV file with correct format (see above).

        Returns
        -------
        calibration : pd.DataFrame
            DataFrame with Multi-Index.

        """
        try:
            calibration = pd.read_csv(
                calibration_fpath, index_col=["Primary", "Setting"]
            )
            calibration.columns = calibration.columns.astype("int64")
            calibration.columns.name = "Wavelength"
            return calibration

        except Exception as e:
            print(cls.load_calibration_file.__doc__)
            raise e

    # Utils

    def _melt_spds(self, spds: pd.DataFrame) -> pd.DataFrame:
        """Convert DataFrame of spds from wide to long format.

        Parameters
        ----------
        spds : pd.DataFrame
            DataFarme of spectral measurements.

        Returns
        -------
        pd.DataFrame
            Long format spds.

        """
        return spds.reset_index().melt(
            id_vars=["Primary", "Setting"],
            value_name="Flux",
            var_name="Wavelength (nm)",
        )

    # Error checking
    def _assert_primary_input_is_valid(
        self, primary: int, primary_input: PrimaryInput
    ) -> None:
        """Check primary input is sensible."""
        if primary not in self.primaries:
            raise StimulationDeviceError(
                f"Requested primary {primary} is unavailable for "
                f"{self.name}. The following primaries are available: "
                f"{self.primaries}"
            )

        if not isinstance(
            primary_input, (int, np.integer, float, np.floating)
        ):
            raise TypeError(
                "Must specify float (>=0.|<=1.) or int (>=0|<=max resolution) "
                + "for primary input."
            )

        msg = (
            f"Requested input {primary_input} exceeds "
            f"resolution of device primary {primary}"
        )

        if isinstance(primary_input, (float, np.floating)):
            if not (primary_input >= 0.0 and primary_input <= 1.0):
                raise StimulationDeviceError(msg)

        elif isinstance(primary_input, (int, np.integer)):
            if not (
                primary_input >= 0
                and primary_input <= self.primary_resolutions[primary]
            ):
                raise StimulationDeviceError(msg)

    def _assert_multiprimary_input_is_valid(
        self, multiprimary_input: DeviceInput
    ) -> None:
        """Check device input is sensible."""
        if len(multiprimary_input) != self.nprimaries:
            raise StimulationDeviceError(
                "Number of primary inputs does not match number of device ",
                "primaries.",
            )

        for primary, primary_input in enumerate(multiprimary_input):
            self._assert_primary_input_is_valid(primary, primary_input)

    def _assert_wavelengths_are_valid(
        self, wavelengths: Sequence[int | float]
    ) -> None:
        """Check wavelengths match those in the calibration file."""
        if not all(
            [wl in self.calibration.columns for wl in np.arange(*wavelengths)]
        ):
            raise StimulationDeviceError(
                "Specfied wavelengths do not match calibration spds."
            )

    def _assert_resolutions_are_valid(
        self, resolutions: Sequence[int]
    ) -> None:
        """Check resolutions are valid."""
        if not len(resolutions) == self.nprimaries:
            raise StimulationDeviceError(
                f"The device has {self.nprimaries} primaries, but "
                + f"{len(resolutions)} resolutions were given."
            )

    def _assert_is_valid_rgb(self, rgb: Sequence[float]) -> None:
        """Check RGB input is valid."""
        all_float = all([isinstance(val, (float, np.floating)) for val in rgb])
        all_between_0_1 = all([(val >= 0.0 and val <= 1.0) for val in rgb])
        all_int = all([isinstance(val, (int, np.integer)) for val in rgb])
        all_between_0_255 = all([(val >= 0 and val <= 255) for val in rgb])

        if not (
            (all_float and all_between_0_1) or (all_int and all_between_0_255)
        ):
            raise StimulationDeviceError(
                "Invalid RGB specification. Please use tuples of float "
                + "(>=0.|<=1.) or int (>=0|<=255)."
            )

    def _assert_colors_are_valid(self, colors: PrimaryColors) -> None:
        """Check primary color specification is valid."""
        if not isinstance(colors, (list, tuple)):
            raise StimulationDeviceError(
                "Must pass a list or tuple of colors."
            )

        if not len(colors) == self.nprimaries:
            raise StimulationDeviceError(
                f"The device has {self.nprimaries} primaries, but "
                + f"{len(colors)} colors were specified."
            )

        if all([isinstance(c, str) for c in colors]):
            mpl_colornames = list(mpl.colors.cnames.keys())
            msg = (
                f" Valid color names: {mpl_colornames}\n\nAt least one color "
                + "name was not valid. Please choose from the above list, or "
                + "pass an array of rgb values."
            )
            if not all([c in mpl_colornames for c in colors]):
                raise ValueError(msg)

        elif all([isinstance(c, (tuple, list)) for c in colors]):
            for rgb in colors:
                self._assert_is_valid_rgb(rgb)

        else:
            raise StimulationDeviceError(
                "Colors must be a list of valid color names or RGB values."
            )

    def _assert_observer_is_valid(self, observer) -> None:
        """Check observer is valid."""
        if not (
            (observer == "CIE_standard_observer")
            or (isinstance(observer, observers._Observer))
        ):
            raise ValueError(
                f"> {observer} is not a valid argument for observer."
            )

    def _get_gamut(self) -> pd.DataFrame:
        """Get xy chromaticity coordinates of the primaries at maximum.

        Returns
        -------
        xy : pd.DataFrame
            Chromaticity coordinates. The xy values for the first primary are
            repeated at the end to complete the path for plotting.

        """
        #breakpoint()
        max_spds = self.calibration.loc[
            (slice(None), self.primary_resolutions), :
        ]
        xyY = max_spds.apply(
            colorfuncs.spd_to_xyY,
            args=(self.calibration_wavelengths[-1],),
            axis=1,
        )
        # Join the dots
        #xyY = pd.concat([xyY, xyY.iloc[0].to_frame().T], ignore_index=True)
        return scipy.spatial.ConvexHull(xyY[["x", "y"]])

    def _xy_in_gamut(self, xy_coord: tuple[float, float]) -> bool:
        """Return True if xy_coord is within the gamut of the device."""
        poly_path = mpl.path.Path(self._get_gamut().points)
        return poly_path.contains_point(xy_coord)

    # TODO: review below here
    # Plotting functions
    def plot_gamut(
        self,
        ax: plt.Axes = None,
        show_1931_horseshoe: bool = True,
        show_CIE170_2_horseshoe: bool = True,
        **kwargs,
    ) -> plt.Figure | None:
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
            colour.plotting.plot_chromaticity_diagram_CIE1931(
                axes=ax, title=False, standalone=False
            )
        if show_CIE170_2_horseshoe:
            cie170_2 = CIE.get_CIE170_2_chromaticity_coordinates()
            cie_horse = ax.plot(
                cie170_2["x"], cie170_2["y"], c="k", ls=":", label="CIE 170-2"
            )

        # Defaults plot kws
        lw = kwargs.pop("lw", 2)
        marker = kwargs.pop("marker", "x")
        markersize = kwargs.pop("markersize", 36)
        color = kwargs.pop("color", "k")
        label = kwargs.pop("label", "Gamut")
        
        # Plot gamut
        for simplex in gamut.simplices:
            lines = ax.plot(
                gamut.points[simplex, 0],
                gamut.points[simplex, 1], 
                lw=lw, color=color, label='Gamut',
                **kwargs
                )
            
        # Plot primary chromaticity coordinates    
        points = ax.scatter(
            gamut.points[:, 0],
            gamut.points[:, 1],
            color=color,
            marker=marker,
            s=markersize,
            label='Primary',
            **kwargs,
        )
        ax.set(xlim=(-0.15, 0.9), ylim=(-0.1, 1), title=f"{self.name}")
        # Fix legend
        handles = [cie_horse[0], lines[0], points]
        ax.legend(handles, ['CIE 170-2', 'Gamut', 'Primaries'])
        return ax

    # TODO: don't require seaborn
    def plot_calibration_spds(
        self, ax: plt.Axes = None, norm: bool = False, **kwargs
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

        # Plot defaults
        lw = kwargs.pop("lw", 0.5)
        legend = kwargs.pop("legend", True)

        # Plot spds
        for primary, color in enumerate(self.primary_colors):
            self.calibration.loc[primary].T.plot(
                c=color, lw=lw, ax=ax, legend=False
            )

        # Add legend
        if legend:
            handles = [
                plt.Line2D([0], [0], c=color) for color in self.primary_colors
            ]
            labels = list(range(self.nprimaries))
    
            # Legend and labels
            ax.legend(handles, labels, title="Primary")
            
        ax.set_title(f"{self.name} SPDs")
        ax.set_ylabel(ylabel)

        return ax

    def plot_calibration_spds_and_gamut(
        self, spd_kwargs: dict | None = None, gamut_kwargs: dict | None = None
    ) -> plt.Figure:
        """Plot calibration SPDs alongside gamut on CIE chromaticity horseshoe.

        Returns
        -------
        fig : plt.Figure
            The figure.

        """
        if spd_kwargs is None:
            spd_kwargs = {}
        if gamut_kwargs is None:
            gamut_kwargs = {}

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        self.plot_calibration_spds(ax=ax1, **spd_kwargs)
        self.plot_gamut(ax=ax2, **gamut_kwargs)
        return fig

    # Predict ouput
    def predict_primary_spd(
        self,
        primary: int,
        primary_input: PrimaryInput,
        name: int | str = 0,
    ) -> npt.NDArray:
        """Predict output for a single device primary at a given setting.

        This is the basis for all predictions.

        Parameters
        ----------
        primary : int
            Device primary.
        primary_input : int or float
            Requested input for device primary. Must be float (>=0.0|<=1.0) or
            int (>=0|<=max resolution).
        name : int or str, optional
            A name for the spectrum. The default is 0.

        Raises
        ------
        ValueError
            If requested value of primary_input exceeds resolution.

        Returns
        -------
        pd.Series
            Predicted spd for primary / primary_input.

        """
        self._assert_primary_input_is_valid(primary, primary_input)
        if isinstance(primary_input, (float, np.floating)):
            # NOTE: Required for smoothness assumption of SLSQP! i.e.,
            # don't use an int.
            primary_input = primary_input * self.primary_resolutions[primary]

        # TODO: these could go in a dispatch table
        f = scipy.interpolate.interp1d(
            x=self.calibration.loc[primary].index.values,
            y=self.calibration.loc[primary],
            axis=0,
            fill_value="extrapolate",
        )
        return pd.Series(
            f(primary_input), name=name, index=self.calibration.columns
        )

    def predict_multiprimary_spd(
        self,
        multiprimary_input: DeviceInput,
        name: int | str = 0,
        nosum: bool = False,
    ) -> pd.Series:
        """Predict spectral power distribution for requested primary inputs.

        Predict the SPD output of the stimulation device for a given list of
        primary inputs. Assumes linear summation of primaries.

        Parameters
        ----------
        multiprimary_input : list of int or float
            Rquested inputs for the device primaries. List of length
            `self.nprimaries` consisting of either float (>=0.0|<=1.0) or
            int (>=0|<=max resolution).
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
        pd.DataFrame if nosum=True else pd.Series
            Predicted spectra or spectrum for given device settings.

        """
        self._assert_multiprimary_input_is_valid(multiprimary_input)
        if name is None:
            name = 0
        spd = []
        for primary, primary_input in enumerate(multiprimary_input):
            spd.append(
                self.predict_primary_spd(primary, primary_input, name=primary)
            )
        spd = pd.concat(spd, axis=1)
        if nosum:
            spd.columns.name = "Primary"
            return spd
        else:
            spd = spd.sum(axis=1)
            spd.name = name
            return spd

    def calculate_aopic_irradiances(self) -> pd.DataFrame:
        """Calculate aopic irradiances from calibration spds.

        Using the CIE026 spectral sensetivities, calculate alpha-opic
        irradiances (S, M, L, R, I) for every spectrum in `self.calibration`.

        Returns
        -------
        pd.DataFrame
            Alphaopic irradiances.

        """
        return self.calibration.dot(self.observer.action_spectra)

    def calculate_lux(self) -> pd.DataFrame:
        """Using the CIE1924 photopic luminosity function, calculate lux for
        every spectrum in `self.calibration`.

        Returns
        -------
        pd.DataFrame
            Lux values.

        """
        vl = CIE.get_CIE_1924_photopic_vl(
            binwidth=self.calibration_wavelengths[-1]
        )
        lux = self.calibration.dot(vl.values) * 683  # lux conversion factor
        lux.columns = ["lux"]
        return lux

    def predict_multiprimary_aopic(
        self,
        multiprimary_input: DeviceInput,
        name: int | str = 0,
    ) -> pd.Series:
        """Predict a-opic irradiances of device for requested primary inputs.

        Parameters
        ----------
        multiprimary_input : list of int or float
            Rquested inputs for the device primaries. List of length
            `self.nprimaries` consisting of either float (>=0.0|<=1.0) or
            int (>=0|<=max resolution).
        name : int or str, optional
            A name for the output, e.g. 'Background'. The default is 0.

        Returns
        -------
        aopic : pd.DataFrame
            Predicted a-opic irradiances for given device settings.

        """
        self._assert_multiprimary_input_is_valid(multiprimary_input)
        spd = self.predict_multiprimary_spd(multiprimary_input, name=name)
        return spd.dot(self.observer.action_spectra)

    def find_settings_xyY(
        self,
        xy: Sequence[float],
        luminance: float,
        tolerance: float = 1e-6,
        plot_solution: bool = False,
        verbose: bool = True,
    ) -> scipy.optimize.OptimizeResult:
        """Find device settings for a spectrum with requested xyY values.

        Parameters
        ----------
        xy : sequence of float
            Requested chromaticity coordinates (xy).
        luminance : float
            Requested luminance.
        tolerance : float, optional
            Acceptable precision for result.
        plot_solution : bool, optional
            Set to True to plot the solution. The default is False.
        verbose : bool, optional
            Set to True to print status messages. The default is False.

        Raises
        ------
        ValueError if len(xy) != 2

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

        requested_xyY = colorfuncs.xy_luminance_to_xyY(xy, luminance)
        requested_LMS = colorfuncs.xyY_to_LMS(requested_xyY)

        # Objective function to find device settings for given xyY
        def _xyY_objective_function(x0: list[float]):
            aopic = self.predict_multiprimary_aopic(x0)
            return sum(
                pow(requested_LMS - aopic[["lc", "mc", "sc"]].to_numpy(), 2)
            )

        # Bounds
        bounds = [
            (
                0.0,
                1.0,
            )
            for primary in self.primary_resolutions
        ]
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
        result = scipy.optimize.basinhopping(
            func=_xyY_objective_function,
            x0=x0,
            niter=100,
            T=1.0,
            stepsize=0.5,
            minimizer_kwargs=minimizer_kwargs,
            callback=_callback,
            interval=50,
            disp=True,
        )

        # TODO: refactor this
        solution_lms = self.predict_multiprimary_aopic(result.x)[
            ["lc", "mc", "sc"]
        ].values
        solution_xyY = colorfuncs.LMS_to_xyY(solution_lms)
        print(f"Requested LMS: {requested_LMS}")
        print(f"Solution LMS: {solution_lms}")

        # Reacfactor!
        if plot_solution is not None:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            # Plot the spectrum
            self.predict_multiprimary_spd(
                result.x, name=f"solution_xyY:\n{solution_xyY.round(3)}"
            ).plot(ax=axs[0], legend=True, c="k")
            self.predict_multiprimary_spd(
                result.x,
                name=f"solution_xyY:\n{solution_xyY.round(3)}",
                nosum=True,
            ).plot(ax=axs[0], color=self.primary_colors, legend=False)
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
            self.plot_gamut(ax=axs[1], show_CIE170_2_horseshoe=True)
            #axs[1].legend()
            device_ao = self.predict_multiprimary_aopic(
                result.x, name="Background"
            )
            colors = [
                val[1] for val in self.observer.photoreceptor_colors.items()
            ]
            device_ao.plot(kind="bar", color=colors, ax=axs[2])

        return result

    def do_gamma(
        self,
        fit: str = "polynomial",
        force_origin: bool = False,
        poly_order: int = 7,
    ) -> None:
        """Make a gamma table from reverse polynomial or beta cdf curve fits.

        For each primary, calculate the unweigthed sum of the spectral
        calibration measurements (output) at each setting (input). A reverse
        curve fit on these data gives the new input values required for
        linearisation.

        Parameters
        ----------
        fit : str, optional
            'polynomial' or 'beta_cdf'. The default is 'polynomial'.
        save_plots_to : str, optional
            Folder to save plots in. The default is None.

        Returns
        -------
        None.

        """
        if not fit in ["polynomial", "beta_cdf"]:
            raise ValueError('Must specify "polynomial" or "beta_cdf" for fit')

        gamma_table = []
        for primary, df in self.calibration.groupby("Primary"):
            settings = df.index.get_level_values("Setting").to_numpy()
            x = settings / settings.max()
            y = df.sum(axis=1).to_numpy()
            y = y / y.max()

            # Reverse fit
            if fit == "polynomial":
                # TODO: can force 0 intercept by passing weights to the fit
                if force_origin:
                    w = np.ones(len(x))
                    w[0] = 1000
                    poly_coeff = np.polyfit(y, x, deg=poly_order, w=w)
                else:
                    poly_coeff = np.polyfit(y, x, deg=poly_order)

                new_x = np.polyval(poly_coeff, x)

            elif fit == "beta_cdf":
                popt, pcov = scipy.optimize.curve_fit(
                    scipy.stats.beta.cdf, y, x, p0=[2.0, 1.0]
                )
                new_x = scipy.stats.beta.cdf(x, *popt)

            new_x = (new_x * self.primary_resolutions[primary]).astype("int")

            # Interpolate for full lookup table
            gamma_table.append(new_x)

        gamma_table = (
            pd.DataFrame(gamma_table, columns=settings)
            .reindex(columns=range(0, max(self.primary_resolutions) + 1))
            .interpolate("linear", axis=1)
            .astype("int")
        )

        # TODO temp fix, make generic
        gamma_table = gamma_table.clip(0, 4095)
        self.gamma = gamma_table

        return None

    def plot_gamma(
        self, save_plots_to: str = None, show_corrected: bool = False
    ) -> None:
        """Plot gamma results for each primary.

        Parameters
        ----------
        save_plots_to : str, optional
            Path to folder where plots should be saved. The default is None.
        show_corrected : bool, optional
            Whether to show gamma corrected predicted values in the plots. The
            default is False.

        Returns
        -------
        None.

        """
        # breakpoint()
        if self.gamma is None:
            raise StimulationDeviceError(
                "Gamma table does not exist. Run .do_gamma() first."
            )

        for primary, df in self.calibration.groupby("Primary"):
            settings = df.index.get_level_values("Setting").to_numpy()
            y = df.sum(axis=1).to_numpy()
            g = self.gamma.loc[primary, settings]
            x = settings / settings.max()
            g = g / g.max()
            y = y / y.max()

            fig, ax = plt.subplots(figsize=(3, 3))
            ax.plot(x, y, c=self.primary_colors[primary], label="Measured")
            ax.plot(x, x, c="k", ls=":", label="Ideal")
            ax.plot(x, g, c="k", label="Reverse fit")
            ax.set_aspect("equal")
            ax.set_xlabel("Input")
            ax.set_ylabel("Output")
            ax.set_title(f"{self.name}")

            if show_corrected:
                corrected = [
                    self.predict_primary_spd(
                        primary, self.gamma_lookup(primary, s)
                    ).sum()
                    for s in settings
                ]
                corrected = corrected / max(corrected)
                ax.plot(x, corrected, c="darkgoldenrod", label="Corrected")

            ax.legend(title=f"Primary {primary}")

            if save_plots_to is not None:
                fig.savefig(
                    op.join(
                        save_plots_to,
                        f'{self.config["json_name"]}_gamma_primary_{primary}.svg',
                    )
                )

        return None

    def gamma_lookup(self, primary: int, primary_input: PrimaryInput) -> int:
        """Look up the gamma corrected value for a primary input.

        Parameters
        ----------
        primary : int
            Device primary.
        primary_input : int or float
            Requested input for device primary. Must be float (>=0.0|<=1.0) or
            int (>=0|<=max resolution).

        Returns
        -------
        int
            Gamma corrected primary input value.

        """
        if self.gamma is None:
            raise StimulationDeviceError(
                "Gamma table does not exist. Run `do_gamma()` first."
            )

        self._assert_primary_input_is_valid(primary, primary_input)
        if isinstance(primary_input, (float, np.floating)):
            primary_input = int(
                primary_input * self.primary_resolutions[primary]
            )
        return int(self.gamma.loc[primary, primary_input])

    def gamma_correct(self, multiprimary_input: DeviceInput) -> list:
        """Return the gamma corrected values for multiprimary device input.

        Parameters
        ----------
        multiprimary_input : Sequence of int or float
            Rquested inputs for the device primaries. List of length
            `self.nprimaries` consisting of either float (>=0.0|<=1.0) or
            int (>=0|<=max resolution).

        Returns
        -------
        list
            Gamma corrected multiprimary input values.

        """
        self._assert_multiprimary_input_is_valid(multiprimary_input)
        return [
            self.gamma_lookup(primary, primary_input)
            for primary, primary_input in enumerate(multiprimary_input)
        ]

    def s2w(self, settings: PrimarySettings) -> list[float]:
        """Convert settings to weights.

        Parameters
        ----------
        settings : Sequence of int
            List of length `self.nprimaries` containing device settings
            ranging from 0 to max-resolution for each primary.

        Returns
        -------
        list of float
            List of primary weights.

        """
        self._assert_multiprimary_input_is_valid(settings)
        if not all(isinstance(s, (int, np.integer)) for s in settings):
            raise StimulationDeviceError(
                "`.s2w()` expects int (>=0|<=max_resolution)."
            )
        return [
            float(s / r) for s, r in zip(settings, self.primary_resolutions)
        ]

    def w2s(self, weights: PrimaryWeights) -> list[int]:
        """Convert weights to settings.

        Parameters
        ----------
        weights : Sequence of float
            List of length `self.nprimaries` containing primary input weights,
            ranging from 0. to 1., for each primary.

        Returns
        -------
        list of int
            List of primary settings.

        """
        self._assert_multiprimary_input_is_valid(weights)
        if not all(isinstance(w, (float, np.floating)) for w in weights):
            raise StimulationDeviceError("`.w2s()` expects float (>=0.|<=1.).")
        return [int(w * r) for w, r in zip(weights, self.primary_resolutions)]
