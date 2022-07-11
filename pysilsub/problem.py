#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pysilsub.problem
================

Condition a silent substitution problem for a given stimulation device and
then solve it with linear algebra or numerical optimisation.

@author: jtm, ms

"""

from typing import List, Optional, Tuple, Sequence

import numpy as np
from scipy.optimize import minimize
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pysilsub.device import StimulationDevice
from pysilsub.colorfunc import LMS_to_xyY, spd_to_xyY, LUX_FACTOR
from pysilsub.plotting import ss_solution_plot
from pysilsub.exceptions import SilSubProblemError


class SilentSubstitutionProblem(StimulationDevice):
    """Class to perform silent substitution with a stimulation device.

    The class inherits from ``pysilsub.device.StimulationDevice`` and is 
    extended with properties and methods to define and solve a silent
    substitution problem. After instantiating the class, set the properties to 
    condition the problem and then use the solver methods to find a solution. 
    For example::

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
        fig = problem.plot_ss_result(solution)


    """

    # Class attibutes
    # Retinal photoreceptors.
    receptors = ["S", "M", "L", "R", "I"]

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
        """Class to perform silent substitution.

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
        config : dict
            Config file dict.

        Returns
        -------
        None.

        """
        # Instance attributes
        super().__init__(
            resolutions,
            colors,
            calibration,
            wavelengths,
            action_spectra,
            name,
            config,
        )

        # Problem parameters, below defined as class properties
        self._background = None
        self._bounds = None
        self._ignore = None
        self._minimize = None
        self._modulate = None
        self._target_contrast = None

        # Default bounds
        self.bounds = [(0.0, 1.0,)] * self.nprimaries

    def __str__(self):
        return f"""SilentSubstitutionProblem:
        Resolutions: {self.resolutions},
        Colors: {self.colors},
        Calibration: {self.calibration.shape},
        Wavelengths: {self.wavelengths},
        Action Spectra: {self.action_spectra.shape},
        Name: {self.name},
        Config: {self.config}
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

            problem.background = [.5] * problem.nprimaries

        Raises
        ------
        SilSubProblemError if requested but not specified.

        """
        if self._background is None:
            print(self.__class__.background.__doc__)
            raise SilSubProblemError(
                "The *background* property has not been set (see above)."
            )
        return self._background

    @background.setter
    def background(self, background_spectrum):
        """Set a background spectrum for the silent substitution problem."""
        if background_spectrum is None:
            self._background = background_spectrum
        else:
            if self._device_input_is_valid(background_spectrum):
                if all([isinstance(i, int) for i in background_spectrum]):
                    self._background = self.settings_to_weights(
                        background_spectrum)
                else:
                    self._background = background_spectrum

    @property
    def bounds(self):
        """Set the bounds for each primary.

        The *bounds* property ensures that solutions are within gamut. Bounds 
        should be defined as a list of tuples of length ``self.nprimaries``, 
        where each tuple consists of a min-max pair defining the lower and 
        upper bounds for each channel. Bounds are set automatically in the 
        ``__init__`` as follows::

            self._bounds = [(0., 1.,)] * self.nprimaries

        If, after creating a problem you want to constrain solutions to avoid
        the lower and upper extremes of the input ranges, for example, you
        could pass something like::

            problem.bounds = [(.02, .98,)] * problem.nprimaries

        It is also possible to set different bounds for each primary.

        Raises
        ------
        SilSubProblemError if requested but not specified.

        """
        if self._bounds is None:
            print(self.__class__.bounds.__doc__)
            raise SilSubProblemError(
                "The *bounds* property has not been set (see above)."
            )
        return self._bounds

    @bounds.setter
    def bounds(self, new_bounds):
        """Set new input bounds for the primaries."""
        if self._bounds_are_valid(new_bounds):
            self._bounds = new_bounds

    @property
    def ignore(self):
        """Specify which photoreceptors to ignore.

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

        """
        if self._ignore is None:
            print(self.__class__.ignore.__doc__)
            raise SilSubProblemError(
                "The *ignore* property has not been set (see above)."
            )
        return self._ignore

    @ignore.setter
    def ignore(self, ignore):
        if self._receptor_input_is_valid(ignore):
            self._ignore = [r for r in self.receptors if r in ignore]

    # TODO: Add [None] ability for minimize
    @property
    def minimize(self):
        """Specify which photoreceptors to minimize.

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

        """
        if self._minimize is None:
            print(self.__class__.minimize.__doc__)
            raise SilSubProblemError(
                "The *minimize* property has not been set (see above)."
            )
        return self._minimize

    @minimize.setter
    def minimize(self, minimize):
        if self._receptor_input_is_valid(minimize):
            self._minimize = [r for r in self.receptors if r in minimize]

    @property
    def modulate(self):
        """Specify which photoreceptor(s) to modulate.

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

        """
        if self._modulate is None:
            print(self.__class__.modulate.__doc__)
            raise SilSubProblemError(
                "The *modulate* property has not been set (see above)."
            )
        return self._modulate

    @modulate.setter
    def modulate(self, modulate):
        if self._receptor_input_is_valid(modulate):
            self._modulate = [r for r in self.receptors if r in modulate]

    @property
    def target_contrast(self):
        """Specify target contrast.

        Setting the *target_contrast* property conditions the silent 
        substitution problem so the solvers know what target contrast to aim 
        for::

            problem.target_contrast = .5

        This would aim for 50% contrast on the photoreceptor(s) that are being
        modulated. If modulating multiple photoreceptors in different 
        directions and / or with different contrasts, target_contrast should be 
        a list of values::

            problem.ignore = ['R']
            problem.minimize = ['I']
            problem.modulate = ['S', 'M', 'L']
            problem.target_contrast = [-.5, .5, .5]

        The above example defines a problem that will ignore rods and aim for
        50% negative contrast on S-cones and 50% positive contrast on L- and 
        M-cones, relative to the background spectrum. 

        Setting *target_contrast* is required for ``linalg_solve()``. When
        using ``optim_solve()``, if target_contrast is not specified then the 
        optimiser will aim to maximise contrast.

        Raises
        ------
        SilSubProblemError if requested but not specified.

        """
        if self._target_contrast is None:
            print(self.__class__.target_contrast.__doc__)
            raise SilSubProblemError(
                "The *target_contrast* property has not been set (see above)."
            )
        return self._target_contrast
    
    # TODO: Add 'max+' and 'max-' options
    @target_contrast.setter
    def target_contrast(self, target_contrast):
        if any((self.ignore is None, self.modulate is None, self.minimize is None)):
            raise SilentSubstitutionProblem(
                'Must specify which photoreceptors to *ignore*, *minimize* and '
                '*isolate* before target_contrast.')
        if target_contrast is None:
            self._target_contrast = None
        else:
            self._target_contrast = np.array(target_contrast)

    # Error checking
    # ---------------
    def _bounds_are_valid(self, new_bounds):
        """Check bounds are valid."""
        if not len(new_bounds) == self.nprimaries:
            print(self.__class__.bounds.__doc__)
            raise SilSubProblemError("Invalid input for bounds (see above).")

        if not all(
            [
                isinstance(item, tuple) and isinstance(b, float)
                for item in new_bounds
                for b in item
            ]
        ):
            print(self.__class__.bounds.__doc__)
            raise SilSubProblemError("Invalid input for bounds (see above).")

        if not all([b[0] >= 0.0 and b[1] <= 1.0 for b in new_bounds]):
            print(self.__class__.bounds.__doc__)
            raise SilSubProblemError("Invalid input for bounds (see above).")

        return True

    def _receptor_input_is_valid(self, receptors):
        """Throw an error if user tries to invent new photoreceptors"""
        if not isinstance(receptors, list):
            raise SilSubProblemError("Please put photoreceptors in a list.")

        if receptors == [None]:
            return True

        else:
            if not all([pr in self.photoreceptors for pr in receptors]):
                raise SilSubProblemError(
                    "Unknown photoreceptor input. Available photoreceptors "
                    "are S-cones, M-cones, L-cones, Rods, and iPRGC's: "
                    f"{self.photoreceptors}"
                )

            if len(set(receptors)) < len(receptors):
                raise SilSubProblemError(
                    "Found duplicate entries for photoreceptors."
                )

        return True

    def _problem_is_valid(self):
        """Check minimal conditions for problem are met."""
        if self.ignore is None:
            print(self.__class__.ignore.__doc__)
            raise SilSubProblemError(
                "The *ignore* property has not been set (see above)."
            )

        if self.minimize is None:
            print(self.__class__.minimize.__doc__)
            raise SilSubProblemError(
                "The *minimize* property has not been set (see above)."
            )

        if self.modulate is None:
            print(self.__class__.modulate.__doc__)
            raise SilSubProblemError(
                "The *modulate* property has not been set (see above)."
            )

        specified = self.ignore + self.minimize + self.modulate
        if not all(
            [pr in specified for pr in self.photoreceptors if pr is not None]
        ):
            print(self.__class__.ignore.__doc__)
            print(self.__class__.minimize.__doc__)
            print(self.__class__.modulate.__doc__)
            raise SilSubProblemError(
                "At least one photoreceptor is not accounted for (see above)."
            )

        return True

    def print_problem(self):
        if self._problem_is_valid():
            print(
                f"{'*'*60}\n{' Silent Substitution Problem ':*^60s}\n{'*'*60}"
                )
            print(f"Device: {self.name}")
            print(f"Background: {self._background}")
            print(f"Ignoring: {self._ignore}")
            print(f"Minimising: {self._minimize}")
            print(f"Modulating: {self._modulate}")
            print(f"Target contrast: {self._target_contrast}")

    # Useful stuff now
    def print_photoreceptor_contrasts(
        self, solution, contrast_statistic="simple"
    ):
        c = self.get_photoreceptor_contrasts(solution, contrast_statistic)
        print(c.round(6))

    def initial_guess_x0(self) -> np.array:
        """Return an initial guess for the optimization variables.

        Returns
        -------
        np.array
            Initial guess for optimization.

        """
        if self._background is not None:
            return np.array(
                [np.random.uniform(lb, ub) for lb, ub in self.bounds]
            )
        else:
            return np.array(
                [np.random.uniform(lb, ub) for lb, ub in self.bounds * 2]
            )

    def smlri_calculator(self, x0: Sequence[float]) -> Tuple[pd.Series]:
        """Calculate alphaopic irradiances for optimisation vector.

        Parameters
        ----------
        x0 : Sequence[float]
            Optimization vector.

        Returns
        -------
        bg_smlri : pd.Series
            Alphaopic irradiances for the background spectrum.
        mod_smlri : pd.Series
            Alphaopic irradiances for the modulation spectrum.

        """
        if self._background == None:
            bg_weights = x0[0: self.nprimaries]
            mod_weights = x0[self.nprimaries: self.nprimaries * 2]
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

    def get_photoreceptor_contrasts(
        self, x0: Sequence[float], contrast_statistic: str = "simple"
    ) -> pd.Series:
        """Return contrasts for ignored, minimized and modulated photoreceptors.

        Parameters
        ----------
        x0 : Sequence[float]
            Optimization vector.
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
        else:
            max_smlri = pd.concat([bg_smlri, mod_smlri], axis=1).max(axis=1)
            min_smlri = pd.concat([bg_smlri, mod_smlri], axis=1).min(axis=1)
            if contrast_statistic == "weber":
                return (max_smlri - min_smlri) / (max_smlri)
            elif contrast_statistic == "michelson":
                return (max_smlri - min_smlri) / (max_smlri + min_smlri)

    # TODO: fix objective function for all use cases
    # Bundle objectives / constraints in a separate class and inherit?
    def objective_function(self, x0: Sequence[float]) -> float:
        """Calculates negative melanopsin contrast for background
        and modulation spectra. We want to minimise this."""
        #breakpoint()
        contrast = self.get_photoreceptor_contrasts(x0)
        if self._target_contrast is not None:
            # Target contrast is specified, aim for target contrast
            return sum(pow(self._target_contrast - contrast[self.modulate], 2))
        
        elif self._target_contrast is None:
            # Target contrast not specified, aim for maximum contrast
            return -sum(contrast[self.modulate])

    # Works fine
    def silencing_constraint(self, x0: Sequence[float]) -> float:
        """Calculates irradiance contrast for minimized photoreceptors.

        """
        #breakpoint()
        contrast = self.get_photoreceptor_contrasts(x0)
        return sum(pow(contrast[self.minimize].values, 2))

    # TODO: add local and global options
    def optim_solve(self):
        """Use optimisation to solve the current silent substitution problem. 

        This method is good for finding maximum available contrast. It uses
        SciPy's ``minimize`` function with the `SLSQP` (sequential quadratic
        least squares programming) solver.

        If ``self.background`` has not been set, both the background and the 
        modulation spectrum will be optimised. 

        If ``self.target_contrast`` has not been set, the optimser will aim to
        maximise contrast, otherwise it will aim to converge on the target
        contrast.

        Returns
        -------
        result : scipy.optimize.OptimizationResult
            The result

        """
        #breakpoint()
        if self._problem_is_valid():
            if self._background is None:
                bounds = self.bounds * 2
                print("> No background specified, will optimise background.")
            else:
                bounds = self.bounds
    
            if self._target_contrast is None:
                print("> No target contrast specified, will aim to maximise.")
    
            print("\n")
    
            constraints = [
                {"type": "eq", "fun": self.silencing_constraint, "tol": 1e-04}
            ]
            
            x0 = self.initial_guess_x0()
            
            result = minimize(
                fun=self.objective_function,
                x0=x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                tol=1e-05,
                options={"maxiter": 100, "ftol": 1e-06, "iprint": 2, "disp": True},
            )
    
            return result

    # TODO: adjust target_contrasts in case of multiple receptor modulate
    # Linear algebra

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
        # breakpoint()
        self._problem_is_valid()
        if self._background is None:
            print(self.__class__.background.__doc__)
            raise SilSubProblemError(
                "Must specify a background spectrum for linalg_solve "
                + "(see above)."
            )

        if self._target_contrast is None:
            print(self.__class__.target_contrast.__doc__)
            raise SilSubProblemError(
                "Must specify target_contrast for linalg_solve "
                + "(see above)."
            )

        # Get a copy of the list of photoreceptors
        receptors = self.receptors.copy()
        for r in self.ignore:
            if r is not None:
                receptors.remove(r)

        # Get only the action spectra we need
        sss = self.action_spectra[receptors]
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

        print(f"receptors: {receptors}")
        print(f"requested: {requested_contrasts}")

        if all([s > b[0] and s < b[1] for s in solution for b in self.bounds]):
            return solution
        else:
            raise ValueError(
                "Solution is out of gamut, lower target contrast.")
            
    def plot_solution(self, x0):
        """Plot the silent substitution solution."""
        # get aopic
        bg_ao, mod_ao = self.smlri_calculator(x0)
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
            mod_spd = self.predict_multiprimary_spd(x0, name="Modulation")
        except:
            bg_spd = self.predict_multiprimary_spd(
                x0[: self.nprimaries], name="Background"
            )
            mod_spd = self.predict_multiprimary_spd(
                x0[self.nprimaries: self.nprimaries * 2], name="Modulation"
            )

        # Print contrasts
        # print(f'\t{self.get_photoreceptor_contrasts(x0)}')

        # get xy
        bg_xy = spd_to_xyY(bg_spd, binwidth=self.wavelengths[2])[:2]
        mod_xy = spd_to_xyY(mod_spd, binwidth=self.wavelengths[2])[:2]

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
