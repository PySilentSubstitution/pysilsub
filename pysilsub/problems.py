"""
pysilsub.problems
=================

Module with interface to definine, solve and visualise silent substitution
problems for a given stimulation device.

@author: jtm, ms

"""
import os.path as op
from typing import List, Optional, Tuple, Sequence, Union

import numpy as np
import numpy.typing as npt
from scipy import optimize
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation

from .devices import (
    StimulationDevice,
    PrimaryWeights,
    DeviceInput,
    PrimaryBounds,
    PrimaryColors,
)
from .colorfuncs import spd_to_xyY
from .plotting import ss_solution_plot
from .observers import _Observer


class SilSubProblemError(Exception):
    """Generic Python-exception-derived object.

    Raised programmatically by ``SilentSubstitutionProblem`` methods when
    arguments are incorrectly specified or when they do not agree with values
    of other .
    """


class SilentSubstitutionProblem(StimulationDevice):
    """Class to perform silent substitution with a stimulation device.

    The class inherits from ``pysilsub.device.StimulationDevice`` and is
    extended with properties and methods to define and solve a silent
    substitution problem. After instantiating, set the properties to define a
    problem and then use the solver methods to find a solution::

        from pysilsub.problem import SilentSubstitutionProblem as SSP

        # EXAMPLE: 1
        # Using all channels at half-power as the background spectrum,
        # use linear algebra to find a solution that gives 20 % S-cone
        # contrast while ignoring rods and silencing M-cones, L-cones
        # and melanopsin.
        problem = SSP.from_package_data('STLAB_York')
        problem.ignore = ['R']
        problem.modulate = ['S']
        problem.minimize = ['M', 'L', 'I']
        problem.background = [.5] * problem.nprimaries
        problem.target_contrast = .2
        solution = problem.linalg_solve()
        fig = problem.plot_ss_result(solution)

        # EXAMPLE: 2
        # Without specifying a background spectrum, use numerical
        # optimisation to find a pair of spectra which maximise
        # melanopsin contrast while ignoring rods and silencing the
        # cones.
        problem = SSP.from_package_data('STLAB_York')
        problem.ignore = ['R']
        problem.modulate = ['I']
        problem.minimize = ['S', 'M', 'L']
        problem.background = None
        problem.target_contrast = None
        solution = problem.optim_solve()
        fig = problem.plot_ss_result(solution.x)


    """

    def __init__(
        self,
        calibration: Union[str, pd.DataFrame],
        calibration_wavelengths: Sequence[int],
        primary_resolutions: Sequence[int],
        primary_colors: PrimaryColors,
        observer: Optional[Union[str, _Observer]] = "CIE_standard_observer",
        name: Optional[str] = None,
        config: Optional[dict] = None,
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

        # Problem parameters
        self._background = None
        self._ignore = None
        self._minimize = None
        self._modulate = None
        self._target_contrast = None

        # Primary bounds
        self._bounds = [
            (0.0, 1.0),
        ] * self.nprimaries

    def __str__(self):
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
        """The background spectrum for the silent substitution stimulus.

        Setting the *background* property internally conditions the silent
        substitution problem.

            * ``linalg_solve(...)`` only works when a background is specified.
            * ``optim_solve(...)`` optimises the background and modulation
              spectrum if no background is specified, otherwise it only
              optimises the modulation spectrum.

        For example, to use all channels at half power as the background
        spectrum::

            problem.background = [.5] * problem.nprimaries

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
    def background(self, background_spectrum: DeviceInput) -> None:
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
    def bounds(self) -> PrimaryBounds:
        """The input bounds for each primary.

        The *bounds* property ensures that solutions are within gamut. Bounds
        should be defined as a list of tuples of length ``self.nprimaries``,
        where each tuple consists of a min-max pair of floats ``(>=0.|<=1.)``
        defining the lower and upper bounds for each channel. Bounds are set
        automatically in the ``__init__`` as follows::

            self._bounds = [(0., 1.,)] * self.nprimaries

        If, after creating a problem you want to constrain solutions to avoid
        the lower and upper extremes of the input ranges, for example, you
        could pass something like::

            problem.bounds = [(.05, .95,)] * problem.nprimaries

        It is also possible to set different bounds for each primary. This may
        be useful if you want to *pin* a problematic primary so that its value
        is constrained during optimisation::

            new_bounds = problem.bounds
            new_bounds[4] = (.49, .51,)
            problem.bounds = new_bounds

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
    def bounds(self, new_bounds: PrimaryBounds) -> None:
        """Set new input bounds for the primaries."""
        self._assert_bounds_are_valid(new_bounds)
        self._bounds = list(new_bounds)

    @property
    def ignore(self) -> Union[List[str], List[None]]:
        """Photoreceptor(s) to be ignored.

        Setting the *ignore* property internally conditions the silent
        substitution problem such that the specified photoreceptor(s) will be
        ignored by the solvers (enabling them to find more contrast). In most
        cases *ignore* will be set to ['R'], because it is considered safe
        practice to ignore rods when stimuli are in the photopic range
        (>300 cd/m2)::

            problem.ignore = ['R']

        But you could also choose to ignore rods and melanopsin::

            problem.ignore = ['R', 'I']

        In the event that you don't want to ignore any photoreceptors, you must
        still pass::

            problem.ignore = [None]

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
    def ignore(self, receptors_to_ignore: Union[List[str], List[None]]) -> None:
        """Set the *ignore* property."""
        self._assert_receptor_input_is_valid(receptors_to_ignore)
        self._ignore = [
            r for r in self.observer.photoreceptors if r in receptors_to_ignore
        ]

    @property
    def minimize(self) -> List[str]:
        """Photoreceptor(s) to be minimized.

        Setting the *minimize* property internally conditions the silent
        substitution problem such that contrast on the target photoreceptors
        will be minimized. For example::

            problem.minimize = ['S', 'M', 'L']

        Would minimize contrast on the cone photoreceptors.

        Setting the *minimize* property is an essential step for conditioning a
        silent substitution problem.

        Raises
        ------
        SilSubProblemError if requested but not specified.

        Returns
        -------
        minimize : list of str
            Photoreceptor(s) to be minimized.

        """
        if self._minimize is None:
            print(self.__class__.minimize.__doc__)
            raise SilSubProblemError(
                "The *minimize* property has not been set (see above)."
            )
        return self._minimize

    @minimize.setter
    def minimize(self, receptors_to_minimize: Sequence[str]) -> None:
        """Set the *minimize* property."""
        self._assert_receptor_input_is_valid(receptors_to_minimize)
        self._minimize = [
            r
            for r in self.observer.photoreceptors
            if r in receptors_to_minimize
        ]

    @property
    def modulate(self) -> List[str]:
        """Photoreceptor(s) to modulate.

        Setting the *modulate* property internally conditions the silent
        substitution problem so the solvers know which photoreceptor(s) should
        be targeted with contrast. For example, to modulate melanopsin::

            problem.modulate = ['I']

        It is also possible to modulate multiple photoreceptors::

            problem.modulate = ['S', 'M', 'L']

        Setting the *modulate* property is an essential step for conditioning a
        silent substitution problem.

        Raises
        ------
        SilSubProblemError if requested but not specified.

        Returns
        -------
        modulate : list of str
            Photoreceptors to be modulated.

        """
        if self._modulate is None:
            print(self.__class__.modulate.__doc__)
            raise SilSubProblemError(
                "The *modulate* property has not been set (see above)."
            )
        return self._modulate

    @modulate.setter
    def modulate(self, receptors_to_modulate: List[str]) -> None:
        """Set the *modulate* property."""
        self._assert_receptor_input_is_valid(receptors_to_modulate)
        self._modulate = [
            r
            for r in self.observer.photoreceptors
            if r in receptors_to_modulate
        ]

    @property
    def target_contrast(self) -> Union[npt.NDArray, str]:
        """Target contrast to aim for on modulated photoreceptor(s).

        Setting the *target_contrast* property conditions the silent
        substitution problem so the solvers know what target contrast to aim
        for::

            problem.target_contrast = .5

        This would aim for 50% contrast on the modulated photoreceptor(s). If
        modulating multiple photoreceptors in different directions and / or
        with different contrasts, *target_contrast* should be a sequence of
        values (if you specify multiple receptors to modulate and pass a float
        then it is assumed *target_contrast* should be the same for each
        photoreceptor)::

            problem.ignore = ['R']
            problem.minimize = ['I']
            problem.modulate = ['S', 'M', 'L']
            problem.target_contrast = [-.5, .5, .5]

        The above example defines a problem that will ignore rods and aim for
        50% negative contrast on S-cones and 50% positive contrast on L- and
        M-cones, relative to the background spectrum.

        Setting *target_contrast* is required for ``linalg_solve()``.

        When using ``.optim_solve()``, *target_contrast* can be set to 'max'
        or 'min', in which case the optimizer will aim to maximize or minimize
        contrast on the modulated photoreceptors.

        Raises
        ------
        SilSubProblemError if requested but not specified.

        Returns
        -------
        target_contrast : NDArray or str
            Contrast to aim for on photoreceptor(s) being modulated.

        """
        if self._target_contrast is None:
            print(self.__class__.target_contrast.__doc__)
            raise SilSubProblemError(
                "The *target_contrast* property has not been set (see above)."
            )
        return self._target_contrast

    @target_contrast.setter
    def target_contrast(
        self, target_contrast: Union[float, Sequence[float], str]
    ) -> None:
        """Set the *target_contrast* property."""
        try:
            self._assert_problem_is_valid()
        except:
            raise SilSubProblemError(
                "Must specify which photoreceptors to *ignore*, *minimize* "
                "and *modulate* before setting *target_contrast*."
            )

        msg = "target_contrast must be float or list of float."
        if target_contrast in ["max", np.inf]:
            self._target_contrast = [np.inf]
        elif target_contrast in ["min", -np.inf]:
            self._target_contrast = [-np.inf]

        elif isinstance(target_contrast, (float, np.floating)):
            # Assume the same for each modulated photoreceptor when float
            self._target_contrast = np.repeat(
                target_contrast, len(self._modulate)
            )
        elif isinstance(target_contrast, list):
            if not len(target_contrast) == len(self._modulate):
                raise SilSubProblemError(
                    f"{len(self._modulate)} receptors are being modulated, "
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

    def _assert_bounds_are_valid(self, new_bounds: PrimaryBounds) -> None:
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
        self, receptors: Sequence[Union[str, None]]
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
        """A problem is valid when all photoreceptors are accounted for by the
        *ignore*, *minimize* and *modulate* properties."""
        # Will throw error if any of these are not set
        specified = self.ignore + self.minimize + self.modulate
        if not all(
            [
                pr in specified
                for pr in self.observer.photoreceptors
                if pr is not None
            ]
        ):
            raise SilSubProblemError(
                "At least one photoreceptor is not accounted for by the "
                + "*ignore*, *minimize* and *modulate* properties."
            )

    def print_problem(self) -> None:
        """Print the current problem definition."""
        self._assert_problem_is_valid()
        print(f"{'*'*60}\n{' Silent Substitution Problem ':*^60s}\n{'*'*60}")
        print(f"Device: {self.name}")
        print(f"Ignoring: {self._ignore}")
        print(f"Minimising: {self._minimize}")
        print(f"Modulating: {self._modulate}")
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

    def smlri_calculator(
        self, x0: PrimaryWeights
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate alphaopic irradiances for candidate solution.

        Parameters
        ----------
        x0 : Sequence of float
            Primary input weights.

        Returns
        -------
        bg_smlri : pd.Series
            Alphaopic irradiances for the background spectrum.
        mod_smlri : pd.Series
            Alphaopic irradiances for the modulation spectrum.

        """
        if self._background is None:
            bg_weights = x0[0 : self.nprimaries]
            mod_weights = x0[self.nprimaries : self.nprimaries * 2]
        else:
            bg_weights = self._background
            mod_weights = x0

        bg_smlri = self.predict_multiprimary_aopic(
            bg_weights, name="Background"
        )
        mod_smlri = self.predict_multiprimary_aopic(
            mod_weights, name="Modulation"
        )
        return (
            bg_smlri,
            mod_smlri,
        )

    def print_photoreceptor_contrasts(
        self, x0: PrimaryWeights, contrast_statistic: Optional[str] = "simple"
    ):
        """Print photoreceptor contrasts for a given solution."""
        contrasts = self.get_photoreceptor_contrasts(x0, contrast_statistic)
        print(contrasts.round(6))

    # TODO: Catch error when background is not set
    def get_photoreceptor_contrasts(
        self, x0: PrimaryWeights, contrast_statistic: Optional[str] = "simple"
    ) -> pd.Series:
        """Return contrasts for ignored, minimized and modulated photoreceptors.

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
        bg_smlri, mod_smlri = self.smlri_calculator(x0)

        if contrast_statistic == "simple":
            return mod_smlri.sub(bg_smlri).div(bg_smlri)

        max_smlri = pd.concat([bg_smlri, mod_smlri], axis=1).max(axis=1)
        min_smlri = pd.concat([bg_smlri, mod_smlri], axis=1).min(axis=1)
        if contrast_statistic == "weber":
            return (max_smlri - min_smlri) / (max_smlri)
        elif contrast_statistic == "michelson":
            return (max_smlri - min_smlri) / (max_smlri + min_smlri)

    def objective_function(self, x0: PrimaryWeights) -> float:
        """Calculates contrast error on modulated photoreceptor(s) in
        accordance with the problem definition.

        Parameters
        ----------
        x0 : Sequence of float
            Primary input weights.

        Returns
        -------
        float
            Contrast error on modulated photoreceptor(s).

        """
        contrast = self.get_photoreceptor_contrasts(x0)
        if self._target_contrast == [np.inf]:
            function_value = -sum(contrast[self.modulate])

        elif self._target_contrast == [-np.inf]:
            function_value = sum(contrast[self.modulate])
        else:
            # Target contrast is specified, aim for target contrast
            function_value = sum(
                pow(self._target_contrast - contrast[self.modulate], 2)
            )
        return function_value

    def silencing_constraint(self, x0: PrimaryWeights) -> float:
        """Calculates contrast error for minimized photoreceptor(s) in
        accordance with the problem definition.

        Parameters
        ----------
        x0 : Sequence of float
            Primary input weights.

        Returns
        -------
        float
            Contrast error on minimized photoreceptor(s).

        """
        contrast = self.get_photoreceptor_contrasts(x0)
        return sum(pow(contrast[self.minimize].values, 2))
    
    # TODO: Release the KWARGS
    def optim_solve(
        self, global_search: Optional[bool] = False, **kwargs
    ) -> optimize.OptimizeResult:
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
        self._assert_problem_is_valid()
        if self._background is None:
            bounds = self.bounds * 2
            print("> No background specified, will optimise background.")
        else:
            bounds = self.bounds

        if self._target_contrast == [np.inf]:
            print("> Aiming to maximise contrast.")

        elif self._target_contrast == [-np.inf]:
            print("> Aiming to minimize contrast.")

        constraints = [
            {"type": "eq", "fun": self.silencing_constraint, "tol": 1e-04}
        ]
        
        if not global_search:  # Local minimization
            
            default_options = {"iprint": 2, "disp": True}
            options = kwargs.pop('options', default_options)
                
            print("> Performing local optimization with SLSQP.")
            result = optimize.minimize(
                fun=self.objective_function,
                x0=self.initial_guess_x0(),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options=options,
                **kwargs
            )
            
        elif global_search:  # Global minimization
            print(
                "> Performing global optimization with basinhopping and SLSQP"
            )
            
            # Configure global defaults
            disp = kwargs.pop('disp', True)
            # Configure local defaults
            default_minimizer_kwargs = {
                "method": "SLSQP",
                "constraints": constraints,
                "bounds": bounds,
                'options': {'iprint':2, 'disp':True}
                }            
            minimizer_kwargs = kwargs.pop('minimizer_kwargs', default_minimizer_kwargs)
                
            # Do optimization
            result = optimize.basinhopping(
                func=self.objective_function,
                x0=self.initial_guess_x0(),
                minimizer_kwargs=minimizer_kwargs,
                disp=disp,
                **kwargs
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

        # TODO: ATTENTION
        # An array for requested contrasts
        requested_contrasts = np.zeros(len(receptors))

        # Plug target_contrast into the right place
        (target_indices,) = np.where(np.in1d(receptors, self.modulate))
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
            raise ValueError("Solution is out of gamut, lower target contrast.")

    def plot_solution(self, solution):
        """Plot the silent substitution solution.
        
        
        Parameters
        ----------
        solution : array_like
            The solution returned by one of the solvers. 
        
        
        Returns
        -------
        fig : plt.Figure
            
        """
        # get aopic
        bg_ao, mod_ao = self.smlri_calculator(solution)
        df_ao = (
            pd.concat([bg_ao, mod_ao], axis=1)
            .T.melt(
                value_name="aopic", var_name="Photoreceptor", ignore_index=False
            )
            .reset_index()
            .rename(columns={"index": "Spectrum"})
        )

        # get spds
        try:
            bg_spd = self.predict_multiprimary_spd(
                self._background, name="Background"
            )
            mod_spd = self.predict_multiprimary_spd(solution, name="Modulation")
        except:
            bg_spd = self.predict_multiprimary_spd(
                solution[: self.nprimaries], name="Background"
            )
            mod_spd = self.predict_multiprimary_spd(
                solution[self.nprimaries : self.nprimaries * 2], name="Modulation"
            )

        # get xy
        bg_xy = spd_to_xyY(bg_spd, binwidth=self.calibration_wavelengths[2])[:2]
        mod_xy = spd_to_xyY(mod_spd, binwidth=self.calibration_wavelengths[2])[
            :2
        ]

        # Make plot
        fig, axs = ss_solution_plot()

        # SPDs
        bg_spd.plot(ax=axs[0], legend=True)
        mod_spd.plot(ax=axs[0], legend=True)

        try:
            axs[0].set_ylabel(self.config.calibration_units)
        except:
            axs[0].set_ylabel("Power")

        # Chromaticity
        axs[1].scatter(
            x=bg_xy[0],
            y=bg_xy[1],
            s=100,
            marker="o",
            facecolors="none",
            edgecolors="k",
            label="Background",
        )
        axs[1].scatter(
            x=mod_xy[0],
            y=mod_xy[1],
            s=100,
            c="k",
            marker="x",
            label="Modulation",
        )
        axs[1].legend()

        # Aopic
        sns.barplot(
            data=df_ao, x="Photoreceptor", y="aopic", hue="Spectrum", ax=axs[2]
        )
        axs[2].set_ylabel("$a$-opic irradiance")

        return fig

    # TODO: Fix
    def plot_contrast_modulation(self, solutions):
        """Sonmething here


        Parameters
        ----------
        solutions : TYPE
            DESCRIPTION.

        Returns
        -------
        fig : TYPE
            DESCRIPTION.

        """
        fig, ax = plt.subplots(1, 1)

        splatter = [self.get_photoreceptor_contrasts(s) for s in solutions]
        splatter = pd.concat(splatter, axis=1)

        for r, s in splatter.iterrows():
            ax.plot(s, label=r, color=self.observer.photoreceptor_colors[r])

            ax.set_ylabel("Simple contrast")
            ax.set_title(self.name)
            ax.set_xlabel("Time (s)")

        plt.legend()
        return fig

    def animate_solution(self, solutions, save_to=None):
        """Something here


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
            """Something"""
            for line in lines:
                line.set_data([], [])
            return lines

        def animate(i):
            """Something"""
            primaries = self.predict_multiprimary_spd(
                solutions[i], nosum=True
            ).T

            for lnum, line in enumerate(lines):
                # set data for each line separately.
                line.set_data(x, primaries.loc[lnum])

            return lines

        # call the animator.  blit=True means only re-draw the parts that have changed.
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
