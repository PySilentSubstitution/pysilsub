#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pysilsub.problem
================

Condition a silent substitution problem for a given stimulation system, and
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
from pysilsub.plotting import stim_plot


class SilSubProblemError(Exception):
    pass


class SilentSubstitutionProblem(StimulationDevice):
    """Class to perform silent substitution with a stimulation device.

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
            for specifying intensity. This is a list of integers to allow for
            systems where primaries may have different resolution depths.
        colors : list[str]
            List of colors for the primaries.
        calibration : pd.DataFrame
            Spectral measurements to characterise the output of the device.
            Column headers must be wavelengths and each row a spectrum.
            Additional columns are needed to identify the primary/setting. For
            example, 380, ..., 780, primary, setting.
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
        self._target_luminance = None
        self._target_xy = None
        self._ignore = None
        self._silence = None
        self._isolate = None
        self._target_contrast = None

        # Other problem parameters me way need
        self._pin_primaries = None

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
              optimises the modulation.

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
    def background(self, background):
        self._background = background

    @property
    def bounds(self):
        """Set the bounds for each primary.

        The bounds property ensures that solutions are within the gamut of the
        device. The bounds should be a list of tuples of length
        ``self.nprimaries``, where each tuple consists of a min-max pair
        defining the lower and upper bounds for each channel. Bounds are set
        automatically in the ``__init__`` as follows::

            self._bounds = [(0., 1.,)] * self.nprimaries

        If, after creating a problem you want to constrain solutions to avoid
        the lower and upper extremes of the input ranges, for example, you
        could pass something like::

            problem.bounds = [(.2, .98,)] * problem.nprimaries

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
    def bounds(self, bounds):
        self._bounds = bounds

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

        If you don't want to ignore any photoreceptors, you must still pass::

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
        ignore = self._check_receptor_input_valid(ignore)
        self._ignore = ignore

    @property
    def silence(self):
        """Specify which photoreceptors to silence.

        Setting the *silence* property internally conditions the silent
        silent substitution problem such that contrast on the target
        photoreceptors will be minimized. For example::

            problem.silence = ['M', 'L']

        Setting the *silence* property is an essential step for conditioning a
        silent substitution problem.

        Raises
        ------
        SilSubProblemError if requested but not specified.

        """
        if self._silence is None:
            print(self.__class__.silence.__doc__)
            raise SilSubProblemError(
                "The *silence* property has not been set (see above)."
            )
        return self._silence

    @silence.setter
    def silence(self, silence):
        silence = self._check_receptor_input_valid(silence)
        self._silence = silence

    @property
    def isolate(self):
        """Specify which photoreceptors to isolate.

        Setting the *isolate* property internally conditions the silent
        substitution problem so the solvers know which photoreceptor(s) should
        be targeted with contrast. For example, to isolate melanopsin::

            problem.isolate = ['I']

        It is also possible to isolate multiple photoreceptors::

            problem.isolate = ['S', 'M', 'L']

        Setting the *isolate* property is an essential step for conditioning a
        silent substitution problem.

        Raises
        ------
        SilSubProblemError if requested but not specified.

        """
        if self._isolate is None:
            print(self.__class__.isolate.__doc__)
            raise SilSubProblemError(
                "The *isolate* property has not been set (see above)."
            )
        return self._isolate

    @isolate.setter
    def isolate(self, isolate):
        isolate = self._check_receptor_input_valid(isolate)
        self._isolate = isolate

    @property
    def target_contrast(self):
        """Specify target contrast.

        Setting the *target_contrast* property internally conditions the
        silent substitution problem so the solvers know what target contrast to
        aim for. For example::

            problem.target_contrast = 1.0

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
                "The *target_contrast* property has not been set (see above)"
            )
        return self._target_contrast

    @target_contrast.setter
    def target_contrast(self, target_contrast):
        self._target_contrast = target_contrast

    # Error checking
    # TODO
    def _check_requested_settings_valid(self):
        pass

    def _check_receptor_input_valid(self, receptors):
        """Throw an error if user tries to invent new photoreceptors"""
        if not isinstance(receptors, list):
            raise SilSubProblemError("Please put photoreceptors in a list.")

        if receptors == [None]:
            return receptors

        else:
            if not all([pr in self.photoreceptors for pr in receptors]):
                raise SilSubProblemError(
                    "Unknown photoreceptor input. Available photoreceptors "
                    "are S-cones, M-cones, L-cones, Rods, and iPRGC's: "
                    f"{self.photoreceptors}"
                )

            if len(set(receptors)) < len(receptors):
                raise SilSubProblemError(
                    "Found duplicate entries for photoreceptors"
                )

        return [r for r in self.receptors if r in receptors]

    def _check_problem_is_valid(self):
        """Check minimal conditions for problem are met."""
        if self.ignore is None:
            print(self.__class__.ignore.__doc__)
            raise SilSubProblemError(
                "The *ignore* property has not been set (see above)."
            )

        if self.silence is None:
            print(self.__class__.silence.__doc__)
            raise SilSubProblemError(
                "The *silence* property has not been set (see above)."
            )

        if self.isolate is None:
            print(self.__class__.isolate.__doc__)
            raise SilSubProblemError(
                "The *isolate* property has not been set (see above)."
            )

        specified = self.ignore + self.silence + self.isolate
        if not all(
            [pr in specified for pr in self.photoreceptors if pr is not None]
        ):
            print(self.__class__.ignore.__doc__)
            print(self.__class__.silence.__doc__)
            print(self.__class__.isolate.__doc__)
            raise SilSubProblemError(
                "At least one photoreceptor is not accounted for (see above)."
            )

        return True

    def print_problem(self):
        self._check_problem_is_valid()
        print(
            "{}\n{:*^60s}\n{}".format(
                "*" * 60, " " + "Silent Substitution Problem" + " ", "*" * 60
            )
        )
        print(f"Device: {self.name}")
        print(f"Background: {self._background}")
        print(f"Ignoring: {self._ignore}")
        print(f"Silencing: {self._silence}")
        print(f"Isolating: {self._isolate}")
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

    def get_photoreceptor_contrasts(
        self, x0: Sequence[float], contrast_statistic: str = "simple"
    ) -> pd.Series:
        """Return contrasts for ignored, silenced and isolated photoreceptors.

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

    def xy_chromaticity_constraint(self, x0: Sequence[float]) -> float:
        """Constraint for chromaticity of background spectrum.

        """
        bg_smlri, _ = self.smlri_calculator(x0)
        bg_xy = LMS_to_xyY(bg_smlri[["L", "M", "S"]])[:2]
        return sum(self.target_xy - bg_xy)

    def luminance_constraint(self, x0: Sequence[float]) -> float:
        """Constraint for luminance of background spectrum.

        """
        bg_smlri, _ = self.smlri_calculator(x0)
        bg_lum = LMS_to_xyY(bg_smlri[["L", "M", "S"]])[2] * LUX_FACTOR
        return pow(self.target_luminance - bg_lum, 2)

    # Plotting func for call back
    def plot_solution(self, background, modulation, ax=None):
        df = (
            pd.concat([background, modulation], axis=1)
            .T.melt(
                value_name="aopic", var_name="Photoreceptor", ignore_index=False
            )
            .reset_index()
            .rename(columns={"index": "Spectrum"})
        )
        fig, ax = plt.subplots()
        sns.barplot(
            data=df, x="Photoreceptor", y="aopic", hue="Spectrum", ax=ax
        )
        plt.show()

    def plot_ss_result(self, x0):
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
                x0[self.nprimaries : self.nprimaries * 2], name="Modulation"
            )

        # Print contrasts
        # print(f'\t{self.get_photoreceptor_contrasts(x0)}')

        # get xy
        bg_xy = spd_to_xyY(bg_spd, binwidth=self.wavelengths[2])[:2]
        mod_xy = spd_to_xyY(mod_spd, binwidth=self.wavelengths[2])[:2]
        print(f"\tBackground xy: {bg_xy}")
        print(f"\tModulation xy: {mod_xy}")

        # Make plot
        fig, axs = stim_plot()

        # SPDs
        bg_spd.plot(ax=axs[0], legend=True)
        mod_spd.plot(ax=axs[0], legend=True)

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
        return fig

    # Bundle objectives / constraints in a separate class and inherit?
    def objective_function(self, x0: Sequence[float]) -> float:
        """Calculates negative melanopsin contrast for background
        and modulation spectra. We want to minimise this."""
        contrast = self.get_photoreceptor_contrasts(x0)
        try:
            # Target contrast is specified, aim for target contrast
            return sum(pow(self._target_contrast - contrast[self._isolate], 2))
        except:
            # Target contrast not specified, aim for maximum contrast
            return -contrast[self._isolate][0]

    # Works fine
    def silencing_constraint(self, x0: Sequence[float]) -> float:
        """Calculates irradiance contrast for silenced photoreceptors.

        """
        bg_smlri, mod_smlri = self.smlri_calculator(x0)
        contrast = self.get_photoreceptor_contrasts(x0)
        return sum(pow(contrast[self._silence], 2))

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
        self._check_problem_is_valid()
        if self._background is None:
            bounds = self.bounds * 2
            print("> No background specified, will optimise background.")
        else:
            bounds = self.bounds

        if self._target_contrast == None:
            print("> No target contrast specified, will aim to maximise.")

        print("\n")

        constraints = [
            {"type": "eq", "fun": self.silencing_constraint, "tol": 1e-02,}
        ]

        result = minimize(
            fun=self.objective_function,
            x0=self.initial_guess_x0(),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            tol=1e-05,
            options={"maxiter": 100, "ftol": 1e-06, "iprint": 2, "disp": True},
        )

        return result

    # TODO: adjust target_contrasts in case of multiple receptor isolate
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
        breakpoint()
        self._check_problem_is_valid()
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
        # A list of requested contrasts
        requested = np.zeros(len(receptors))
        for i in self.isolate:
            requested[receptors.index(i)] = self.target_contrast

        # Adjust requested values from percentage to native units as a function
        # of the background spectrum (divide by 2?)
        requested = A.sum(axis=1).mul(requested)

        # Get inverse function
        if A.shape[0] == A.shape[1]:  # Square matrix, use inverse
            inverse_function = np.linalg.inv
        else:  # Use pseudo inverse
            inverse_function = np.linalg.pinv

        # Inverse
        A1 = pd.DataFrame(inverse_function(A.values), A.columns, A.index)

        # The solution
        solution = A1.dot(requested) + self.background

        if all([s > b[0] and s < b[1] for s in solution for b in self.bounds]):
            return solution
        else:
            raise ValueError("Solution is out of gamut, lower target contrast.")
