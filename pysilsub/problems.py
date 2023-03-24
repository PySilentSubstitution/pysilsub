"""
``pysilsub.problems``
=====================

Definine, solve and visualise silent substitution problems for a given observer
and multiprimary stimulation device.

"""
from __future__ import annotations


import os.path as op
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import colour

from . import devices
from . import colorfuncs
from . import observers
from . import CIE
from . import waves


class SilSubProblemError(Exception):
    """Generic Python-exception-derived object.

    Raised programmatically by ``SilentSubstitutionProblem`` methods when
    arguments are incorrectly specified or when they do not agree with values
    of other .
    """


class SilentSubstitutionProblem(devices.StimulationDevice):
    """Class to perform silent substitution with a stimulation device.

    The class inherits from ``pysilsub.device.StimulationDevice`` and is
    extended with properties and methods to define and solve a silent
    substitution problem. After instantiating, set the properties to define a
    problem and then use the solver methods to find a solution::

        import pysilsub

        # EXAMPLE: 1
        # Using all channels at half-power as the background spectrum,
        # use linear algebra to find a solution that gives 20 % S-cone
        # contrast while ignoring rods and silencing M-cones, L-cones
        # and melanopsin.
        ssp = pysilsub.problems.from_package_data('STLAB_York')
        ssp.ignore = ['rh']
        ssp.target = ['sc']
        ssp.silence = ['mc', 'lc', 'mel']
        ssp.background = [.5] * ssp.nprimaries
        ssp.target_contrast = .2
        solution = ssp.linalg_solve()
        fig = ssp.plot_ss_result(solution)

        # EXAMPLE: 2
        # Without specifying a background spectrum, use numerical
        # optimisation to find a pair of spectra which maximise
        # melanopsin contrast while ignoring rods and silencing the
        # cones.
        ssp = pysilsub.problems.from_package_data('STLAB_York')
        ssp.ignore = ['rh']
        ssp.target = ['mel']
        ssp.silence = ['sc', 'mc', 'lc']
        ssp.background = None
        ssp.target_contrast = None
        solution = ssp.optim_solve()
        fig = ssp.plot_ss_result(solution.x)


    """

    def __init__(
        self,
        calibration: str | pd.DataFrame,
        calibration_wavelengths: Sequence[int],
        primary_resolutions: Sequence[int],
        primary_colors: devices.PrimaryColors,
        observer: str | observers._Observer = "CIE_standard_observer",
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
        observer : str or pysilsub.observer.Observer, optional
        observer : str or pysilsub.observer.Observer, optional
            Colorimetric observer model. The default is "CIE_standard_observer"
            , in which case defaults to the CIE standard colorimetric observer
            model (age=32, field_size=10).
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
        # Instance attributes
        super().__init__(
            calibration,
            calibration_wavelengths,
            primary_resolutions,
            primary_colors,
            observer,
            name,
            config,
        )

        # TODO: default as [] instead of None
        # Problem parameters
        self._background = None
        self._ignore = None
        self._silence = None
        self._target = None
        self._target_contrast = None

        # Primary bounds
        self._bounds = [
            (0.0, 1.0),
        ] * self.nprimaries

    def __str__(self):
        """Return a string representation of the SilentSubstitutionProblem."""
        return f"""
    SilentSubstitutionProblem:
        Calibration: {self.calibration.shape},
        Calibration Wavelengths: {self.calibration_wavelengths},
        Primary Resolutions: {self.primary_resolutions},
        Primary Colors: {self.primary_colors},
        Observer: {self.observer},
        Name: {self.name}
    """

    # Properties of a SilentSubstitutionProblem
    @property
    def background(self):
        """Set a background spectrum for the silent substitution stimulus.

        Setting the *background* property internally conditions the silent
        substitution problem.

            * ``linalg_solve(...)`` only works when a background is specified.
            * ``optim_solve(...)`` optimises the background and modulation
              spectrum if no background is specified, otherwise it only
              optimises the modulation spectrum.

        For example, to use all channels at half power as the background
        spectrum::

            ssp.background = [.5] * ssp.nprimaries

        Raises
        ------
        SilSubProblemError if requested but not specified.

        Returns
        -------
        background : Sequence of float or int
            Primary inputs of the background spectrum.

        """
        if self._background is None:
            print(self.__class__.background.__doc__)
            raise SilSubProblemError(
                "The *background* property has not been set (see above)."
            )
        return self._background

    @background.setter
    def background(self, background_spectrum: devices.DeviceInput) -> None:
        """Set a background spectrum for the silent substitution problem."""
        if background_spectrum is None:
            self._background = background_spectrum
        else:
            self._assert_multiprimary_input_is_valid(background_spectrum)
            if all(
                [isinstance(i, (int, np.integer)) for i in background_spectrum]
            ):
                self._background = self.s2w(background_spectrum)
            else:
                self._background = background_spectrum

    @property
    def bounds(self) -> devices.PrimaryBounds:
        """Define input bounds for each primary.

        The *bounds* property ensures that solutions are within gamut. Bounds
        should be defined as a list of tuples of length ``self.nprimaries``,
        where each tuple consists of a min-max pair of floats ``(>=0.|<=1.)``
        defining the lower and upper bounds for each channel. Bounds are set
        automatically in the ``__init__`` as follows::

            self._bounds = [(0., 1.,)] * self.nprimaries

        If, after creating a problem you want to constrain solutions to avoid
        the lower and upper extremes of the input ranges (for example) you
        could pass something like::

            ssp.bounds = [(.05, .95,)] * ssp.nprimaries

        It is also possible to set different bounds for each primary. This may
        be useful if you want to *pin* a problematic primary so that its value
        is constrained during optimisation::

            new_bounds = ssp.bounds
            new_bounds[4] = (.49, .51,)  # Constrain the 4th primary
            ssp.bounds = new_bounds

        Raises
        ------
        SilSubProblemError if requested but not specified.

        Returns
        -------
        bounds : List of float pairs
            Input bounds for each primary.

        """
        if self._bounds is None:
            print(self.__class__.bounds.__doc__)
            raise SilSubProblemError(
                "The *bounds* property has not been set (see above)."
            )
        return self._bounds

    @bounds.setter
    def bounds(self, new_bounds: devices.PrimaryBounds) -> None:
        """Set new input bounds for the primaries."""
        self._assert_bounds_are_valid(new_bounds)
        self._bounds = list(new_bounds)

    @property
    def ignore(self) -> list[str] | list[None]:
        """Photoreceptor(s) to be ignored.

        Setting the *ignore* property internally conditions the silent
        substitution problem such that the specified photoreceptor(s) will be
        ignored by the solvers (enabling them to find more contrast on the
        targeted photoreceptors). For example, one may choose to *ignore* rods
        when working in the photopic range (>300 cd/m2) as they are often
        assumed to be saturated and incapable of signalling at these light
        levels::

            ssp.ignore = ['rh']

        One may also choose to ignore both rods and melanopsin::

            ssp.ignore = ['rh', 'mel']

        Or both rods and the non-functional cone of a dichromat::

            ssp.ignore = ['rh', 'lc']

        In the event that you don't want to ignore any photoreceptors, you must
        still pass::

            ssp.ignore = [None]

        Setting the *ignore* property is an essential step for conditioning a
        silent substitution problem.

        Raises
        ------
        SilSubProblemError if ignore not specified.

        Returns
        -------
        ignore : [None] or list of str
            Photoreceptors to be ignored.

        """
        if self._ignore is None:
            print(self.__class__.ignore.__doc__)
            raise SilSubProblemError(
                "The *ignore* property has not been set (see above)."
            )
        return self._ignore

    @ignore.setter
    def ignore(self, receptors_to_ignore: list[str] | list[None]) -> None:
        """Set the *ignore* property."""
        self._assert_receptor_input_is_valid(receptors_to_ignore)
        self._ignore = [
            r for r in self.observer.photoreceptors if r in receptors_to_ignore
        ]

    @property
    def silence(self) -> list[str]:
        """Photoreceptor(s) to be (nominally) silenced.

        Setting the *silence* property internally conditions the silent
        substitution problem such that contrast on the target photoreceptors
        will be minimized. For example::

            ssp.silence = ['sc', 'mc', 'lc']

        Would aim to silence the cone photoreceptors.

        Setting the *silence* property is an essential step for conditioning a
        silent substitution problem.

        Raises
        ------
        SilSubProblemError if requested but not specified.

        Returns
        -------
        silence : list of str
            Photoreceptor(s) to be silenced.

        """
        if self._silence is None:
            print(self.__class__.silence.__doc__)
            raise SilSubProblemError(
                "The *silence* property has not been set (see above)."
            )
        return self._silence

    @silence.setter
    def silence(self, receptors_to_silence: Sequence[str]) -> None:
        """Set the *silence* property."""
        self._assert_receptor_input_is_valid(receptors_to_silence)
        self._silence = [
            r
            for r in self.observer.photoreceptors
            if r in receptors_to_silence
        ]

    @property
    def target(self) -> list[str]:
        """Photoreceptor(s) to target.

        Setting the *target* property internally conditions the silent
        substitution problem so the solvers know which photoreceptor(s) should
        be targeted with contrast. For example, to target melanopsin::

            ssp.target = ['mel']

        It is also possible to target multiple photoreceptors::

            ssp.target = ['sc', 'mc', 'lc']

        Setting the *target* property is an essential step for conditioning a
        silent substitution problem.

        Raises
        ------
        SilSubProblemError if requested but not specified.

        Returns
        -------
        target : list of str
            Photoreceptors to target.

        """
        if self._target is None:
            print(self.__class__.target.__doc__)
            raise SilSubProblemError(
                "The *target* property has not been set (see above)."
            )
        return self._target

    @target.setter
    def target(self, receptors_to_target: list[str]) -> None:
        """Set the *target* property."""
        self._assert_receptor_input_is_valid(receptors_to_target)
        self._target = [
            r for r in self.observer.photoreceptors if r in receptors_to_target
        ]

    @property
    def target_contrast(self) -> npt.NDArray | str:
        """Target contrast to aim for on targeted photoreceptor(s).

        Setting the *target_contrast* property conditions the silent
        substitution problem so the solvers know what target contrast to aim
        for::

            ssp.target_contrast = .5

        This would aim for 50% contrast on the targeted photoreceptor(s). If
        modulating multiple photoreceptors in different directions and / or
        with different contrasts, *target_contrast* should be a sequence of
        values (if you specify multiple receptors to targeted and pass a float
        then it is assumed *target_contrast* should be the same for each
        photoreceptor)::

            ssp.ignore = ['rh']
            ssp.silence = ['sc', 'mel']
            ssp.target = ['mc', 'lc']
            ssp.target_contrast = [-.5, .5]

        The above example defines a problem that will ignore rods, silence
        S-cones and melanopsin, and aim for 50% negative contrast on M-cones
        and 50% positive contrast on L-cones, relative to the background
        spectrum.

        Note
        ----
        Setting *target_contrast* is required for ``linalg_solve()``.

        When using ``.optim_solve()``, *target_contrast* can be set to 'max'
        or 'min', in which case the optimizer will aim to maximize or minimize
        contrast on the targeted photoreceptors.

        Raises
        ------
        SilSubProblemError if requested but not specified.

        Returns
        -------
        target_contrast : NDArray or str
            Contrast to aim for on photoreceptor(s) being targeted.

        """
        if self._target_contrast is None:
            print(self.__class__.target_contrast.__doc__)
            raise SilSubProblemError(
                "The *target_contrast* property has not been set (see above)."
            )
        return self._target_contrast

    @target_contrast.setter
    def target_contrast(
        self, target_contrast: float | Sequence[float] | str
    ) -> None:
        """Set the *target_contrast* property."""
        try:
            self._assert_problem_is_valid()
        except Exception:
            raise SilSubProblemError(
                "Must specify which photoreceptors to *ignore*, *silence* "
                "and *target* before setting *target_contrast*."
            )

        msg = "target_contrast must be float or list of float."
        if target_contrast in ["max", np.inf]:
            self._target_contrast = [np.inf]
        elif target_contrast in ["min", -np.inf]:
            self._target_contrast = [-np.inf]

        elif isinstance(target_contrast, (float, np.floating)):
            # Assume the same for each targeted photoreceptor when float
            self._target_contrast = np.repeat(
                target_contrast, len(self._target)
            )
        elif isinstance(target_contrast, list):
            if not len(target_contrast) == len(self._target):
                raise SilSubProblemError(
                    f"{len(self._target)} receptors are being target, "
                    + f"but {len(target_contrast)} target_contrasts given."
                )
            if not all(
                [
                    isinstance(val, (float, np.floating))
                    for val in target_contrast
                ]
            ):
                raise SilSubProblemError(msg)
            self._target_contrast = np.array(target_contrast)
        else:
            raise SilSubProblemError(msg)

    # Error checking
    # ---------------

    def _assert_bounds_are_valid(
        self, new_bounds: devices.PrimaryBounds
    ) -> None:
        """Check bounds are valid."""
        correct_length = len(new_bounds) == self.nprimaries
        tuples_of_float = all(
            [
                isinstance(item, tuple) and isinstance(b, (float, np.floating))
                for item in new_bounds
                for b in item
            ]
        )
        in_gamut = all([b[0] >= 0.0 and b[1] <= 1.0 for b in new_bounds])

        if not all([correct_length, tuples_of_float, in_gamut]):
            print(self.__class__.bounds.__doc__)
            raise SilSubProblemError("Invalid input for bounds (see above).")

    def _assert_receptor_input_is_valid(
        self, receptors: Sequence[str | None]
    ) -> None:
        """Throw an error if user tries to invent new photoreceptors."""
        if not isinstance(receptors, (list, tuple)):
            raise SilSubProblemError(
                "Please put photoreceptors in a list or tuple."
            )

        if receptors != [None]:
            if not all(
                [pr in self.observer.photoreceptors for pr in receptors]
            ):
                raise SilSubProblemError(
                    "Unknown photoreceptor input. Available photoreceptors "
                    "are S-cones, M-cones, L-cones, Rods, and iPRGC's: "
                    f"{self.observer.photoreceptors}"
                )

            if len(set(receptors)) < len(receptors):
                raise SilSubProblemError(
                    "Found duplicate entries for photoreceptors."
                )

    def _assert_problem_is_valid(self) -> None:
        """Check the problem is valid.

        A problem is valid when all photoreceptors are accounted for by the
        *ignore*, *silence* and *target* properties.
        """
        # Will throw error if any of these are not set
        specified = self.ignore + self.silence + self.target
        if not all(
            [
                pr in specified
                for pr in self.observer.photoreceptors
                if pr is not None
            ]
        ):
            raise SilSubProblemError(
                "At least one photoreceptor is not accounted for by the "
                + "*ignore*, *silence* and *target* properties."
            )

    def print_problem(self) -> None:
        """Print the current problem definition."""
        self._assert_problem_is_valid()
        print(f"{'*'*60}\n{' Silent Substitution Problem ':*^60s}\n{'*'*60}")
        print(f"Device: {self.name}")
        print(f"Observer: {self.observer}")
        print(f"Ignoring: {self._ignore}")
        print(f"Minimising: {self._silence}")
        print(f"Modulating: {self._target}")
        print(f"Target contrast: {self._target_contrast}")
        print(f"Background: {self._background}")
        print("\n")

    # Useful stuff now
    def initial_guess_x0(self) -> npt.NDArray:
        """Return an initial guess of the primary inputs for optimization.

        Returns
        -------
        NDArray
            Initial guess for optimization.

        """
        if self._background is not None:
            return np.array(
                [np.random.uniform(lb, ub) for lb, ub in self.bounds]
            )
        return np.array(
            [np.random.uniform(lb, ub) for lb, ub in self.bounds * 2]
        )

    def aopic_calculator(
        self, x0: devices.PrimaryWeights
    ) -> tuple[pd.Series, pd.Series]:
        """Calculate alphaopic irradiances for candidate solution.

        Parameters
        ----------
        x0 : Sequence of float
            Primary input weights.

        Returns
        -------
        bg_aopic : pd.Series
            Alphaopic irradiances for the background spectrum.
        mod_aopic : pd.Series
            Alphaopic irradiances for the modulation spectrum.

        """
        if self._background is None:
            bg_weights = x0[0 : self.nprimaries]
            mod_weights = x0[self.nprimaries : self.nprimaries * 2]
        else:
            bg_weights = self._background
            mod_weights = x0

        bg_aopic = self.predict_multiprimary_aopic(
            bg_weights, name="Background"
        )
        mod_aopic = self.predict_multiprimary_aopic(
            mod_weights, name="Modulation"
        )
        return (
            bg_aopic,
            mod_aopic,
        )

    def print_photoreceptor_contrasts(
        self, x0: devices.PrimaryWeights, contrast_statistic: str = "simple"
    ):
        """Print photoreceptor contrasts for a given solution."""
        contrasts = self.get_photoreceptor_contrasts(x0, contrast_statistic)
        print(contrasts.round(6))

    # TODO: Catch error when background is not set
    # TODO: When background is set, do we need michelson?
    def get_photoreceptor_contrasts(
        self, x0: devices.PrimaryWeights, contrast_statistic: str = "simple"
    ) -> pd.Series:
        """Return contrasts for ignored, silenced and targeted photoreceptors.

        Parameters
        ----------
        x0 : Sequence of float
            Primary input weights.
        contrast_statistic : str
            The contrast statistic to return, either 'simple', 'weber', or
            'michelson'.

        Returns
        -------
        pd.Series
            Photoreceptor contrasts.

        """
        bg_aopic, mod_aopic = self.aopic_calculator(x0)

        if contrast_statistic == "simple":
            return mod_aopic.sub(bg_aopic).div(bg_aopic)

        max_aopic = pd.concat([bg_aopic, mod_aopic], axis=1).max(axis=1)
        min_aopic = pd.concat([bg_aopic, mod_aopic], axis=1).min(axis=1)
        if contrast_statistic == "weber":
            return (max_aopic - min_aopic) / (max_aopic)
        elif contrast_statistic == "michelson":
            return (max_aopic - min_aopic) / (max_aopic + min_aopic)

    def objective_function(self, x0: devices.PrimaryWeights) -> float:
        """Calculate contrast error on targeted photoreceptor(s).

        This happens in accordance with the problem definition.

        Parameters
        ----------
        x0 : Sequence of float
            Primary input weights.

        Returns
        -------
        float
            Contrast error on targeted photoreceptor(s).

        """
        contrast = self.get_photoreceptor_contrasts(x0)
        if np.inf in self._target_contrast:
            function_value = -sum(contrast[self.target])

        elif -np.inf in self._target_contrast:
            function_value = sum(contrast[self.target])

        else:
            # Target contrast is specified, aim for target contrast
            function_value = sum(
                pow(self._target_contrast - contrast[self.target], 2)
            )
        return function_value

    def silencing_constraint(self, x0: devices.PrimaryWeights) -> float:
        """Calculate contrast error for silenced photoreceptor(s).

        Parameters
        ----------
        x0 : Sequence of float
            Primary input weights.

        Returns
        -------
        float
            Contrast error on silenced photoreceptor(s).

        """
        contrast = self.get_photoreceptor_contrasts(x0)
        return sum(pow(contrast[self.silence].values, 2))

    # TODO: Release the KWARGS
    # TODO: contrast is coming out at half
    def optim_solve(
        self, x0: devices.PrimaryWeights = None, global_search: bool = False, **kwargs
    ) -> scipy.optimize.OptimizeResult:
        """Use optimisation to solve the current silent substitution problem.

        This method is good for finding maximum available contrast. It uses
        SciPy's `minimize` function with the `SLSQP` (sequential quadratic
        least squares programming) solver.

        If `self.background` has not been set, both the background and the
        modulation spectrum are optimised.

        If `self.target_contrast` has been set to `np.inf` or `-np.inf`, the
        optimser will aim to maximise or minimize contrast, respectively,
        otherwise it will aim to converge on the target contrast.

        Parameters
        ----------
        global_search : bool, optional
            Whether to perform a global search with scipy's basinhopping
            algorithm.

        Returns
        -------
        result : scipy.optimize.OptimizeResult
            The result of the optimization.

        """
        print(f'{" optim_solve ":~^60s}')
        self._assert_problem_is_valid()
        if self._background is None:
            bounds = self.bounds * 2
            print("> No background specified, will optimise background.")
        else:
            bounds = self.bounds

        if np.inf in self._target_contrast:
            print("> Aiming to maximise contrast.")

        elif -np.inf in self._target_contrast:
            print("> Aiming to minimize contrast.")

        constraints = [
            {"type": "eq", "fun": self.silencing_constraint, "tol": 1e-04}
        ]

        if x0 is None:
            x0 = self.initial_guess_x0()
            
        if not global_search:  # Local minimization

            default_options = {"iprint": 2, "disp": True, "ftol": 1e-08}
            options = kwargs.pop("options", default_options)

            print("> Performing local optimization with SLSQP.")
            result = scipy.optimize.minimize(
                fun=self.objective_function,
                x0=x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options=options,
                **kwargs,
            )

        elif global_search:  # Global minimization
            print(
                "> Performing global optimization with basinhopping and SLSQP"
            )

            # Configure global defaults
            disp = kwargs.pop("disp", True)
            # Configure local defaults
            default_minimizer_kwargs = {
                "method": "SLSQP",
                "constraints": constraints,
                "bounds": bounds,
                "options": {"iprint": 2, "disp": False},
            }
            minimizer_kwargs = kwargs.pop(
                "minimizer_kwargs", default_minimizer_kwargs
            )

            # Do optimization
            result = scipy.optimize.basinhopping(
                func=self.objective_function,
                x0=x0,
                minimizer_kwargs=minimizer_kwargs,
                disp=disp,
                **kwargs,
            )

        return result

    def linalg_solve(self):
        """Use linear algebra to solve the current silent substitution problem.

        Inverse method is applied when the matrix is square (i.e., when the
        number of primaries is equal to the number of photoreceptors),
        alternatively the Moore-Penrose pseudo-inverse.

        Raises
        ------
        SilSubProblemError
            If `self.background` or `self.target_contrast` are not specified.
        ValueError
            If the solution is outside `self.bounds`.

        Returns
        -------
        solution : pd.Series
            Primary weights for the modulation spectrum.

        """
        self._assert_problem_is_valid()
        if self._background is None:
            print(self.__class__.background.__doc__)
            raise SilSubProblemError(
                "Must specify a background spectrum for linalg_solve "
                + "(see above)."
            )

        if isinstance(self._target_contrast, str):
            raise SilSubProblemError(
                ".linalg_solve() does not support 'max' or 'min' contrast "
                + "modes. Please specify a target_contrast."
            )

        if self._target_contrast is None:
            print(self.__class__.target_contrast.__doc__)
            raise SilSubProblemError(
                "Must specify target_contrast for linalg_solve "
                + "(see above)."
            )

        # Get a copy of the list of photoreceptors
        receptors = self.observer.photoreceptors.copy()
        for receptor in self.ignore:
            if receptor is not None:
                receptors.remove(receptor)

        # Get only the action spectra we need
        sss = self.observer.action_spectra[receptors]
        bg_spds = self.predict_multiprimary_spd(self.background, nosum=True)

        # Primary to sensor matrix
        A = sss.T.dot(bg_spds)

        # An array for requested contrasts
        requested_contrasts = np.zeros(len(receptors))

        # Plug target_contrast into the right place
        (target_indices,) = np.where(np.in1d(receptors, self.target))
        requested_contrasts[target_indices] = self.target_contrast

        # Scale requested values from percentage to native units as a function
        # of the background spectrum (divide by 2?)
        requested_contrasts = A.sum(axis=1).mul(requested_contrasts)

        # Get inverse function
        if A.shape[0] == A.shape[1]:  # Square matrix, use inverse
            inverse_function = np.linalg.inv
        else:  # Use pseudo inverse
            inverse_function = np.linalg.pinv

        # Inverse
        A1 = pd.DataFrame(inverse_function(A.values), A.columns, A.index)

        # TODO: Why does dividing by two give the expected contrast, and not
        # dividing by two gives roughly double!?
        solution = A1.dot(requested_contrasts) / 2 + self.background

        # print(f"receptors: {receptors}")
        # print(f"requested: {requested_contrasts}")

        if all([s > b[0] and s < b[1] for s in solution for b in self.bounds]):
            return solution
        else:
            raise ValueError(
                "Solution is out of gamut, lower target contrast."
            )

    # Solutions and plotting

    def get_solution_spds(self, solution):
        """Get predicted spds for solution.

        Parameters
        ----------
        solution : array_like
            The solution returned by one of the solvers.

        Returns
        -------
        bg_spd : pd.Series
            Chromaticity coordinates of background spectrum.
        mod_spd : pd.Series
            Chromaticity coordinates of modulation spectrum.

        """
        # get spds
        try:
            bg_spd = self.predict_multiprimary_spd(
                self._background, name="Background"
            )
            mod_spd = self.predict_multiprimary_spd(
                solution, name="Modulation"
            )
        except Exception:
            bg_spd = self.predict_multiprimary_spd(
                solution[: self.nprimaries], name="Background"
            )
            mod_spd = self.predict_multiprimary_spd(
                solution[self.nprimaries : self.nprimaries * 2],
                name="Modulation",
            )
        return bg_spd, mod_spd

    def plot_solution_spds(self, solution, ax=None, **kwargs):
        """Plot the solution spds.

        Parameters
        ----------
        solution : array_like
            The solution returned by one of the solvers.
        ax : plt.Axes, optional
            Axes on which to  plot. The default is None.
        **kwargs : dict
            Passed to `pd.Series.plot()`.

        Returns
        -------
        ax : plt.Axes
            Axes on which the data are plotted.
        """
        bg_spd, mod_spd = self.get_solution_spds(solution)

        # Make plot
        if ax is None:
            ax = plt.gca()

        # SPDs
        bg_spd.plot(ax=ax, legend=True, **kwargs)
        mod_spd.plot(ax=ax, legend=True, **kwargs)

        units = self.config.get("calibration_units")
        if units is None:
            ax.set_ylabel("Power")
        else:
            ax.set_ylabel(units)

        return ax

    def get_solution_xy(self, solution):
        """Get xy chromaticity coordinates for solution spectra.

        Parameters
        ----------
        solution : array_like
            The solution returned by one of the solvers.

        Returns
        -------
        bg_xy : TYPE
            Chromaticity coordinates of background spectrum.
        mod_xy : TYPE
            Chromaticity coordinates of modulation spectrum.
        """
        bg_spd, mod_spd = self.get_solution_spds(solution)

        # get xy
        bg_xy = colorfuncs.spd_to_xyY(
            bg_spd, binwidth=self.calibration_wavelengths[2]
        )[:2]
        mod_xy = colorfuncs.spd_to_xyY(
            mod_spd, binwidth=self.calibration_wavelengths[2]
        )[:2]
        return bg_xy, mod_xy

    def plot_solution_xy(self, solution, ax=None, **kwargs):
        """Plot solution chromaticity coordinates on CIE horseshoe.

        Parameters
        ----------
        solution : array_like
            The solution returned by one of the solvers.
        ax : plt.Axes, optional
            Axes on which to  plot. The default is None.
        **kwargs : dict
            Passed to `pd.Series.scatter()`.

        Returns
        -------
        ax : plt.Axes
            Axes on which the data are plotted.
        """
        # get xy
        bg_xy, mod_xy = self.get_solution_xy(solution)

        # Make plot
        if ax is None:
            ax = plt.gca()

        # Plot solution on horseshoe
        colour.plotting.plot_chromaticity_diagram_CIE1931(
            axes=ax, title=False, standalone=False
        )

        cie170_2 = CIE.get_CIE170_2_chromaticity_coordinates()
        ax.plot(cie170_2["x"], cie170_2["y"], c="k", ls=":", label="CIE 170-2")
        ax.legend()
        ax.set(title="CIE 1931 horseshoe", xlim=(-0.1, 0.9), ylim=(-0.1, 0.9))

        # Chromaticity
        ax.scatter(
            x=bg_xy[0],
            y=bg_xy[1],
            s=100,
            marker="o",
            facecolors="none",
            edgecolors="k",
            label="Background",
        )
        ax.scatter(
            x=mod_xy[0],
            y=mod_xy[1],
            s=100,
            c="k",
            marker="x",
            label="Modulation",
        )
        ax.legend()
        return ax

    def plot_solution_aopic(self, solution, ax=None, **kwargs):
        """Plot barchart of alphaopic irradiances .

        Parameters
        ----------
        solution : array_like
            The solution returned by one of the solvers.
        ax : plt.Axes, optional
            Axes on which to  plot. The default is None.
        **kwargs : dict
            Passed to `DataFrame.bar.plot()`.

        Returns
        -------
        ax : plt.Axes
            Axes on which the data are plotted.
        """
        # Make plot
        if ax is None:
            ax = plt.gca()

        # Default kwargs
        width = kwargs.pop("width", 0.7)

        # Get aopic and plot
        bg_ao, mod_ao = self.aopic_calculator(solution)
        data = pd.concat([bg_ao, mod_ao], axis=1)
        data.plot.bar(ax=ax, rot=0, width=width, **kwargs)

        # Set labels. Some fun bugs solved here :)
        ax.set_ylabel(r"$\alpha$-opic irradiance")
        xlabels = [f"$E_{{{pr}}}$" for pr in self.observer.photoreceptors]
        ax.set_xticklabels(xlabels)
        ax.set_xlabel("")
        return ax

    def plot_solution(self, solution):
        """Plot solution spds, xy chromaticities and alphaopic irradiances.

        Parameters
        ----------
        solution : array_like
            The solution returned by one of the solvers.

        Returns
        -------
        fig : plt.Figure

        """
        # Make plot
        fig, axs = plt.subplots(1, 3, figsize=(13, 4))

        # SPDs
        self.plot_solution_spds(solution, ax=axs[0])
        # xy
        self.plot_solution_xy(solution, ax=axs[1])
        # Aopic
        self.plot_solution_aopic(solution, ax=axs[2])

        return fig

    def print_solution(self, solution):
        """Print device input settings for background and modulation.

        Parameters
        ----------
        solution : array_like
            The solution returned by one of the solvers.

        Returns
        -------
        None.

        """
        if self._background is None:
            bg_weights = solution[0 : self.nprimaries]
            mod_weights = solution[self.nprimaries : self.nprimaries * 2]
        else:
            bg_weights = self._background
            mod_weights = solution

        print(f"Background spectrum: {self.w2s(bg_weights)}")
        print(f"Modulation spectrum: {self.w2s(mod_weights)}")

    # TODO: polish
    def make_contrast_modulation(
        self,
        frequency: int | float,
        sampling_frequency: int,
        target_contrast: float,
        phase: int | float = 0,
        duration: int | float = 1
        ):
        """Make a sinusoidal contrast modulation.

        Parameters
        ----------
        frequency : int | float
            Modulation frequency in Hz.
        sampling_frequency : int
            Sampling frequency of the modulation. This should not exceed the
            temporal resolution (spectral switching time) of the stimulation
            device.
        target_contrast : float
            Peak contrast of the modulation for targetted photoreceptor(s).
        phase : int | float, optional
            Phase offset. The default is 0. Use np.pi
        duration : int | float, optional
            Duration of the modulation. The default is 1.

        Returns
        -------
        solutions : list
            Solution for each time point in the modulation.

        """
        waveform = waves.make_stimulus_waveform(
            frequency,
            sampling_frequency,
            phase,
            duration,
        )

        solutions = []
        for point in waveform:
            contrast = point * target_contrast
            self.target_contrast = contrast
            solutions.append(self.linalg_solve())
        return solutions

    # TODO: Fix
    def plot_contrast_modulation(self, solutions: list, **kwargs):
        """Plot photoreceptor contrast over time for a list of solutions.

        Parameters
        ----------
        solutions : list
            List of solutions returned by one of the solvers (one for each
            timepoint in the modulation).

        Returns
        -------
        fig : plt.Figure
            The plot.

        """
        fig, ax = plt.subplots(1, 1, figsize=(3.54, 3.54))

        # Calculate contrast for all photoreceptors
        contrasts = [self.get_photoreceptor_contrasts(s) for s in solutions]
        contrasts = pd.concat(contrasts, axis=1)

        # Plot
        contrasts.T.plot(ax=ax, color=self.observer.photoreceptor_colors)
        ax.set_ylabel("Contrast")
        ax.set_title(self.name)
        ax.set_xlabel("Time point")

        return fig

    def animate_solutions(self, solutions, save_to=None):
        """Something here.

        Parameters
        ----------
        solutions : TYPE
            DESCRIPTION.
        save_to : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        TYPE
            DESCRIPTION.
        """
        fig = plt.figure()
        ax = plt.axes(xlim=(300, 800), ylim=(0, 2e06))
        ax.set_xlabel("Wavelength (nm)")
        x = np.arange(380, 781, 1)

        lines = []
        for p, c in zip(self.primaries, self.primary_colors):
            (line,) = ax.plot([], [], lw=2, color=c, label=f"Primary {p}")
            lines.append(line)

        ax.legend()

        def init():
            """Something."""
            for line in lines:
                line.set_data([], [])
            return lines

        def animate(i):
            """something."""
            primaries = self.predict_multiprimary_spd(
                solutions[i], nosum=True
            ).T

            for lnum, line in enumerate(lines):
                # set data for each line separately.
                line.set_data(x, primaries.loc[lnum])

            return lines

        # call the animator.  blit=True means only re-draw the parts that
        # have changed.
        anim = animation.FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=len(solutions),
            interval=1,
            blit=True,
        )

        if save_to is not None:
            anim.save(
                op.join(save_to, "ss_animation.mp4"),
                fps=30,
                extra_args=["-vcodec", "libx264"],
            )
            anim.save(
                op.join(save_to, "ss_basic_animation.gif"),
                fps=30,
                writer="imagemagick",
            )

        return anim
