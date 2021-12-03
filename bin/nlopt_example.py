import sys
sys.path.insert(0, '../')

import seaborn as sns
import pandas as pd
import nlopt
from numpy import *


from silentsub.silentsub import SilentSubstitutionProblem
from silentsub.colorfunc import LMS_to_xyY, xyY_to_LMS
from silentsub.plotting import stim_plot

sns.set_context('notebook')
sns.set_style('whitegrid')


spds = pd.read_csv('../data/S2_corrected_oo_spectra.csv', index_col=['led','intensity'])
spds.index.rename(['Primary', 'Setting'], inplace=True)
spds.columns = pd.Int64Index(spds.columns.astype(int))
spds.columns.name = 'Wavelength'


# list of colors for the primaries
colors = ['blueviolet', 'royalblue', 'darkblue', 'blue', 'cyan', 
          'green', 'lime', 'orange', 'red', 'darkred']

ss = SilentSubstitutionProblem(
    resolutions=[4095]*10,
    colors=colors,
    spds=spds,
    spd_binwidth=1,
    isolate=['I'],
    silence=['S', 'M', 'L'],
    target_contrast=1.
    )

target_xy=[.33, .33]
target_luminance=600.

bg = ss.find_settings_xyY(target_xy, target_luminance)
ss.background = bg.x

# Define constraints and local minimizer
constraints = [{
    'type': 'eq',
    'fun': ss.silencing_constraint
}]


# Set up nlopt
lb = zeros(10)
ub = ones(10)
opt = nlopt.opt(nlopt.LD_SLSQP, 10)
opt.set_lower_bounds(lb)
opt.set_upper_bounds(ub)
opt.set_min_objective(lambda x, *args: ss.objective_function(x))
opt.add_equality_constraint(lambda x, *args: ss.silencing_constraint(x))
opt.set_xtol_rel(1e-4)
x = opt.optimize(ss.initial_guess_x0())
minf = opt.last_optimum_value()
print("optimum at ", x[0], x[1])
print("minimum value = ", minf)
print("result code = ", opt.last_optimize_result())


# Plot result
# bg, mod = ss.smlri_calculator(result.x)
# ss.plot_solution(bg, mod)
# ss.debug_callback_plot(result.x)