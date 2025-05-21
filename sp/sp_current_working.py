#! /usr/bin/python
#
##### SURVIVAL PROBABILITY PLOT FOR MOLECULAR DYNAMICS SIMULATIONS ##### 
#              Written by Jakub LICKO, MChem    
#              Adapted from MDAnalysis official documentation (insert ref)   
#
#  INSTRUCTIONS
#  Example command line input:
#       python surv_prob.py -t trajout.xtc -s topol.tpr -ref "name Uo1" -sel "name OW*, name OBT*" -start 45000 -stop -1 -ts 2 -taumax 20 
#  Flags:
#       -t : centred trajectory file
#       -s : topology file (tpr for GROMACS)
#       -ts : time (ps) between frames. default = 2
#       -ref : reference atom to compute SP around
#       -geom : geometrical selection around ref to compute SP around
#       -sel : selection of species, in quotation marks, comma separated selections (e.g. "resname SOL, resname Na")
#       -start : first trajectory frame to analyse
#       -stop : final trajectory frame to analyse (default -1, i.e. last frame)
#       -csv : save info to csv file? 
#       -taumax : number of frames to compute SP for
#

    #            The key aim of this code is to extract the time constant (1/k).
    #            This value gives an indication as to the lifetime of a particular species ("dynamic")
    #            in a particular region - in this case a defined radius around the "static" group.
    #            The time-constant is also known as the MEAN LIFETIME,
    #            and is defined by the time when the survival probability has decreased from 1 to 1/e (~ 0.368).
    #            It is up to the user to know whether this value and interpretation is of use for their system.''')

#
#   Required installed python libraries (includes pre-installed libraries): 
#       numpy
#       matplotlib.pyplot
#       itertools
#       scipy
#       scienceplots (optional: only added here to make more visually appealing plots)
#       MDAnalysis
#       os
#       argparse
#       sys
#       logging
#       time
#       cycler

import numpy as np 
# import matplotlib as mpl
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis.waterdynamics import SurvivalProbability as SP
from scipy.optimize import curve_fit
import argparse
import itertools
import os
import scienceplots
import sys
import logging
from cycler import cycler
from time import time

# parse user input arguments
parser = argparse.ArgumentParser(description="Specify analysis options")
parser.add_argument('-t', help='trajectory file')
parser.add_argument('-s', default='topol.tpr', help='topology file')
# parser.add_argument('-ref', help='which atom to compute a survival probability around')
# parser.add_argument('-geom', help='geometry relative to reference to compute survival probability in. E.g. "sphzone 12.3" ')
# parser.add_argument('-sel', help='species to sample. NOTE: string needs to be in quotation marks, separate selections with commas')
parser.add_argument('-ts', default=2, help='timestep (in ps) BETWEEN FRAMES')
parser.add_argument('-dynamic', help='what is the dynamic group (will be treated together)? e.g. neptunium atoms adsorbed on clay')
parser.add_argument('-static', help='what is the static group (will be treated separately)? e.g. the AT* sites on a clay surface')
parser.add_argument('-intermittency', help='how many frames can the dynamic group leave the geometry for')
parser.add_argument('-radius', help='radius (or cut-off) to calculate SP for')
parser.add_argument('-start', default=0, help='initial frame to read')
parser.add_argument('-stop', default=-1, help='final frame to read')
parser.add_argument('-csv', choices=['yes','no'], default = 'yes', help='Save SP of ligands to csv files? Options: yes (default), no')
parser.add_argument('-taumax', default=20, help='number of frames to compute SP for')
parser.add_argument('-curvefit', default='k', choices=['k','ak','kc','akc'], help='which exp curve to fit for? exp(-kx), exp(-kx)+c, a*exp(-kx), or a*exp(-kx)+c')
parser.add_argument('-constmin', help='MINimum value of time constant for PLOTTING. the value will still be printed into results file')
parser.add_argument('-constmax', help='MAXimum value of time constant for PLOTTING. the value will still be printed into results file')



# parser.add_argument('-csv', choices=['yes','no'], default = 'yes', help='Save positions of selections and substitution sites to csv files? Options: yes (default), no')
args = vars(parser.parse_args())

# convert user inputs into variables to use later
# ref = args['ref']
# sel = args['sel'].split(', ')
traj = args['t']
topol = args['s']
dynamic = args['dynamic']
static = args['static']
# ano2 = args['ano2']
# geom = args['geom']
ts = int(args['ts'])
radius = float(args['radius'])
frame_start = int(args['start'])
frame_stop = int(args['stop'])
taumax = int(args['taumax'])
csv = args['csv']
curvefit = args['curvefit']
constmin = float(args['constmin'])
constmax = float(args['constmax'])
intermittency = args['intermittency']

# logging 
logname = "SP.log"
logger = logging.getLogger("surv_prob")
fh = logging.FileHandler(logname)
ch = logging.StreamHandler()
logger.addHandler(fh)
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)
logger.info(" ".join(sys.argv))
logger.info("")
        
# start time
# start_time = time()

# define universe
u = mda.Universe(topol, traj, in_memory=False)

# find current directory
cwd = os.getcwd()

#################################
# define curve fitting function #
#################################

def round_to_3_digits(value):
    formatstr = '%.' + str(3)+'g'
    return float(formatstr % value)

def surv_prob_curve_fit():
    # data prep - specifies how the SP timeseries data will be used by the function
    x = time_timeseries
    y = sp_timeseries

    # possible parameters to be fitted: 
    # a = pre-exponential factor (optional) 
    # k = decay coefficient 
        # this is the key parameter for comparison between selected species
    # c = constant, serves as the horizontal asymptote (optional)

    # give parameters global scope (so the code can recognise them when the function is called)
    global popt, pcov, perr, a, k, c, x_fitted, y_fitted, cond_numb, time_constant, errors_rounded

    if curvefit == 'ak':
        # define optimization parameters and their covariance coefficients
        
        popt, pcov = curve_fit(lambda t, a, k: a * (np.exp(-k * t)), x, y)

        # define a, k
        a = popt[0]
        k = popt[1]

        # define fitted x and y
        # x_fitted = np.linspace(np.min(x), np.max(x), 100) #why 100?
        y_fitted = a * np.exp(-k * x)

    elif curvefit == 'kc':
        # define optimization parameters and their covariance coefficients
        
        popt, pcov = curve_fit(lambda t, k, c: (np.exp(-k * t)) + c, x, y)

        # define k, c
        k = popt[0]
        k = round_to_3_digits(k)        
        c = popt[1]
        c = round_to_3_digits(c)        


        # define fitted x and y
        # x_fitted = np.linspace(np.min(x), np.max(x), 100) #why 100?
        y_fitted = np.exp(-k * x) + c

    elif curvefit == 'akc':
        # define optimization parameters and their covariance coefficients
        popt, pcov = curve_fit(lambda t, a, k, c: a * (np.exp(-k * t)) + c, x, y)

        # define a, k and c 
        a = popt[0]
        a = round_to_3_digits(a)
        k = popt[1]
        k = round_to_3_digits(k)        
        c = popt[2]
        c = round_to_3_digits(c)        

        # define fitted x and y
        # x_fitted = np.linspace(np.min(x), np.max(x), 100) #again, why 100 again?
        y_fitted = a * np.exp(-k * x) + c

    elif curvefit == 'k':
        # define optimization parameters and their covariance coefficients
        popt, pcov = curve_fit(lambda t, k: (np.exp(-k * t)), x, y)

        # define a, k and c 
        k = popt[0]
        k = round_to_3_digits(k)        

        # define fitted x and y
        # x_fitted = np.linspace(np.min(x), np.max(x), 100) #again, why 100 again?
        y_fitted = np.exp(-k * x)

    # calculate the time constant (1 / k)
    time_constant = 1 / k 
    time_constant = round_to_3_digits(time_constant)
    if np.isinf(time_constant) == True:
        time_constant = 0

    # Additional Stats
    # Calculating the STDEV of each fitted parameter (popt) from the generated covariance matrix (pcov)
    # NOTE: pcov diagonal values are the VARIANCE (sigma^2) values for each popt (off-diagonal terms are covariance values)
    # the code below thus takes the square root of each diagonal term to calculate the Standard Deviation
    perr = np.sqrt(np.diag(pcov))

    errors_rounded = []
    for i in perr:
        i = round_to_3_digits(i)
        errors_rounded.append(i)
    print(errors_rounded)

    # check for fit overparametrization with the Condition Number of the matrix
    cond_numb = np.linalg.cond(pcov)

    # LEGACY plotting code, kept from original curve_fit tutorial (link: HERE)
    # ax = plt.axes()
    # ax.scatter(x, y, label='Raw data')
    # ax.plot(x_fitted, y_fitted, 'k', label='Fitted curve')
    # ax.set_title(r'Using curve_fit() to fit an exponential function')
    # ax.set_ylabel('y-Values')
    # ax.set_ylim(0, 1)
    # ax.set_xlabel('x-Values')
    # ax.legend()

#########################
## create results file ##
#########################

with open("SP_results.txt", "w") as file:
    file.write('\n##############################################')
    file.write('\n# Survival Probability Curve-Fitting Results #')
    file.write('\n##############################################')
    file.write(f"\nCalculated in directory: {cwd}")
    # file.write(f'\nReference: {ref}\nFull selection: ')
    file.write(f'\nDynamic: {dynamic}\nStatic: {static}\nGeometry: around {radius} Ångstrom(s)\nframes: {frame_start} to {frame_stop}\ntau: {taumax}')
    if curvefit == "ak":
        file.write(f'\nCurve fit equation: y = a * exp(-k * x)')
    elif curvefit =='akc':
        file.write(f'\nCurve fit equation: y = a * exp(-k * x) + c')
    elif curvefit =='kc':
        file.write(f'\nCurve fit equation: y = exp(-k * x) + c')
    elif curvefit =='k':
        file.write(f'\nCurve fit equation: y = exp(-k * x)')




############ LEGACY PLOTTING CODE ##############
#### this is from when this code was written to ideally produce plots directly after calculation ####
#### this turned out to not be possible after the code was rewritten to treat static selections individually ####
#### BUT it might be possible if the code is further modified to filter through the SP data somehow ####
#### idea for this future modification: filter through the SP results, and don't plot those that are 1,1,1,1,1... or 1,0,0,0... ####
# plot formatting cycler - line colours and line styles
# default_cycler = (cycler(color=['r', 'g', 'b', 'orange']) +
                #   cycler(linestyle=['-', '--', ':', '-.']))

# define scatter plot colours and markers
# marker = itertools.cycle(('o', '+', 'x', '*'))
# colours = itertools.cycle(("red", "green", "blue", "orange"))


# # make nice plots
plt.style.use(['science','notebook','grid','no-latex'])
# initialise plotting
fig1, ax1 = plt.subplots() # combined figure

# # weirdly, specifying 'no-latex' actually DOES generate plots with LaTeX font, even if it is not installed
# I don't understand why, but it is what it is
###############################################

#########################################
###### Static group atom selection ######
#########################################
# if tetrahedral substitution sites were selected as the static group, this if statement 'filters' this selection...
# ...so that only those exposed to bulk solution are sampled
# this isn't technically necessary, but it just reduces the calculation and the number of results needed to be sifted through
if static == 'name AT*':
    # substitution sites
    #dimensions of simulation box (Sanity check):
    box_dim = u.dimensions
    minX, maxX = 0, box_dim[0]
    minY, maxY = 0, box_dim[1]
    minZ, maxZ = 0, box_dim[2]

    # select clay atoms
    clay = u.select_atoms('resname UC*')

    # create array containing clay positions, with x y and z being in separate rows
    # i.e. transform FROM N atoms with 3 coordinates TO 3 coordinate sets with N entries each
    clay_positions = np.transpose(clay.positions)

    # identify max and min z coordinates of all clay atoms (only max is needed for SDM generation)
    clay_min_z = np.min(clay_positions[2])
    clay_max_z = np.max(clay_positions[2])

    # AT to study
    top_layer = clay_max_z - 2    
    top_at = u.select_atoms(f'name AT* and (prop z <= {clay_max_z} and 'f'prop z >= {top_layer})')
    bottom_layer = clay_min_z + 2    
    bottom_at = u.select_atoms(f'name AT* and (prop z >= {clay_min_z} and 'f'prop z <= {bottom_layer})')

    static_selection = top_at + bottom_at    
else:
    static_selection = u.select_atoms(f'{static}')


#DIRECTLY FROM DOCUMENTATION - added by Lorenz lab group
# I tried implementing this averaging method but it wasn't working due to the way I manipulate my data compared to them
# so my averaging and data storage is different, but somewhat inspired by them
# joined_sp_timeseries = [[] for _ in range(num_of_AT)]

# counter for numbering files and looping
counter = 1

# calculation loop
for static_sel_resid in static_selection.resids:
    #################################################
    ######## SURVIVAL PROBABILITY CALCULATION #######
    #################################################

    import MDAnalysis.transformations as trans
    u2 = mda.Universe(topol, traj, in_memory=False)
    at_site = u2.select_atoms(f'resid {static_sel_resid} and {static}')

    # not_at_site = u2.select_atoms('not (name AT* and resid 1)')

    # # transforms = [trans.unwrap(u2.atoms)]
    # #             # trans.center_in_box(at_site, center='geometry'),
    # #             # trans.wrap(not_at_site)]
    # transforms = [
    #     trans.translate(-at_site.center_of_mass()),   # Move the reference atom to the origin
    #     trans.wrap(at_site, compound='atoms')  # Wrap all atoms into the box relative to the reference atom
    # ]

    def center_atom_in_box(ts):
        # Get box center from dimensions (ts.dimensions[:3] = box lengths in x, y, z)
        box_center = 0.5 * ts.dimensions[:3]
        
        # Compute shift needed to move the reference atom to the box center
        shift_vector = box_center - at_site.center_of_mass()
        
        # Apply the shift
        u2.atoms.translate(shift_vector)

        return ts

    # Add transformations: center the atom in the box, then wrap all atoms
    u2.trajectory.add_transformations(center_atom_in_box, trans.wrap(u2.atoms, compound='atoms'))

    # select reference and selection pair and calculate SP for it
    select = f"{dynamic} and around {radius} (resid {static_sel_resid} and {static})" # I wasn't able to find a different way to select for those AT 
    sp = SP(u2, select, verbose=True)
    sp.run(start=frame_start, stop=frame_stop, tau_max=taumax, intermittency=intermittency)
    tau_timeseries = sp.tau_timeseries

    ######## MY MODIFICATION 1 - TIMESTEP MULTIPLIER
    # original function takes no account for time between frames (i.e. the TIMESTEP (ts))
    # so while tau is defined in terms of time in the original documentation, it isn't always necessarily the case
    # hence, I introduce a TIMESTEP (ts) here - the difference between frames is 2 ps, i.e. 0.002 ns (50 000 frames == 100 000 ps == 100 ns)
    # the value of the TIMESTEP is 2 by default, but it can be defined by the user with the -ts flag
    time_timeseries = [x * ts for x in tau_timeseries]
    time_timeseries = np.array(time_timeseries) #change it from a list into a numpy array
    print(f'time timeseries: {time_timeseries}')
    # END OF MODIFICATION

    # define the surv prob array
    sp_timeseries = sp.sp_timeseries
    sp_timeseries = np.nan_to_num(np.array(sp_timeseries))
    print(f'sp timeseries: {sp_timeseries}')
    

    ####################################
    ####### saving to CSV file #########
    ####################################
    
    if csv == 'yes':
        print(f'Saving {dynamic} values into csv file')
        surv_prob_data = time_timeseries, sp_timeseries
        surv_prob_data = np.transpose(surv_prob_data)
        
        # define filename, replace whitespaces with underscores and asterisks with 'all' 
        csv_filename = f'{dynamic}_surv_prob_{counter}'
        csv_filename = csv_filename.replace(' ','_')
        csv_filename = csv_filename.replace('*','all')

        # save as a csv file 
        np.savetxt(f'{csv_filename}.csv', surv_prob_data, delimiter = ',', header=f'SP timeseries of sel {dynamic}, {radius} of ref {static}, count {counter}\n{cwd}')  
        # with open('sp_data.txt', 'w') as file:
        #     file.write()
    else:
        print('Survival probability not being saved')   
    #   END OF MODIFICATION 2

    ###################################
    ######### FIT TO CURVE ############
    ###################################
    
    #need a try statement here because in some cases (if full of 0s or 1s etc), the curve-fitting function won't work and will give an error
    try:
        # fit curve
        surv_prob_curve_fit()
        
        # print lambda
        print(time_constant)

        # plot combined figure if not ISC or not non-existent
        if abs(time_constant) > constmax or abs(time_constant) < constmin or time_constant == 0: # Set limits for which time constant curves to plot
            pass
        else:
            color = next(ax1._get_lines.prop_cycler)['color']
            ax1.plot(time_timeseries, sp_timeseries, label=f'{counter}', color=color, linewidth=3)
            # plt.plot(time_timeseries, y_fitted, color=color, linestyle = '--', linewidth=1)     

            fig2, ax2 = plt.subplots(figsize=(3,3)) # separate figures

            ax2.plot(time_timeseries, sp_timeseries, label=f'{counter}', color=color, linewidth = 1)
            ax2.plot(time_timeseries, y_fitted, color=color, linestyle = '--', linewidth=4)
            ax2.set_xlabel('Time (ps)')
            ax2.set_ylabel('SP')
            ax2.set_ylim(-0.05,1.05)

            fig2.savefig(f'SP_separate_{counter}.png', dpi=200, bbox_inches = 'tight')


        ##############################
        # Write results to txt file #
        ##############################
        with open('SP_results.txt', 'a') as file:
            file.write('\n--------------------')
            file.write(f'\n{counter}, corresponds to {csv_filename}')

            # file.write(f'\nCurve fit parameters for {dynamic} ({colour} {plot_marker})') # legacy code that included plotting
            file.write(f'\nCurve fit parameters for {dynamic}, static resid: {static_sel_resid})')

            # write curve fit parameters
            if curvefit == 'k':
                file.write(f'\nk = {k}')
            elif curvefit == 'ak': 
                file.write(f'\na = {a}\nk = {k}')
            elif curvefit == 'kc': 
                file.write(f'\nk = {k}\nc = {c}')
            elif curvefit == 'akc':
                file.write(f'\na = {a}\nk = {k}\nc = {c}')

            # write time constant
            file.write(f'\n\nTime constant (1/k) = {time_constant} ps')

            # write curve fit parameter error values
            file.write(f'\n\nCurve fit parameter STDEV values (calculated by taking the square root of covariance matrix diagonal terms):')
            if curvefit == 'k':
                file.write(f'\nk STDEV = {errors_rounded[0]}')
            elif curvefit =='ak':
                file.write(f'\na STDEV = {errors_rounded[0]}\nk STDEV = {errors_rounded[1]}')
            elif curvefit =='kc':
                file.write(f'\nk STDEV = {errors_rounded[0]}\nc STDEV = {errors_rounded[1]}')
            elif curvefit =='akc':
                file.write(f'\na STDEV = {errors_rounded[0]}\nk STDEV = {perr[1]}\nc STDEV = {errors_rounded[2]}')
        
            # write covariance matrix   
            file.write(f'\n\nCovariance matrix:')
            file.write(f'\n{pcov}')
            file.write(f'\n\nCovariance matrix condition number (overfitting check)')
            file.write(f'\n{cond_numb}\n')
    except:
        pass
    counter += 1
    # END OF MODIFICATION 3


########
# IDEA: Add a section either at the start or end of the SP_results.txt file
# listing all the calculated k values. Relevant? Useful?
########

########## MORE LEGACY PLOTTING CODE ###########
# # plotting
# # plt.scatter(time_timeseries, sp_mean, c=colour, marker=plot_marker)
# plt.scatter(time_timeseries, sp_timeseries, c=colour, marker=plot_marker )
# plt.plot(x, y_fitted, c=colour)

# # ORDER IS IMPORTANT - plt.rc(...) must be first, THEN plt.grid()
# # plt.rc('axes', prop_cycle = default_cycler)
# # plt.grid()

ax1.set_xlabel('Time (ps)')
ax1.set_ylabel('SP')
ax1.set_ylim(-0.05,1.05)
ax1.set_title(f'SP - {dynamic} within {radius} {static}')

# #%%
# # PLOT GENERATION AND SAVING     

# # define plot title
plot_title = f'SP_frame{frame_start}to{frame_stop}_tau{taumax}_ref_{dynamic}_combined'

#replace whitespaces with underscores and asterisks with
plot_title = plot_title.replace(' ','_')

#save figure - multiple options for presentations, thesis, publications, etc
fig1.savefig(f'{plot_title}_small.png', bbox_inches='tight')
fig1.savefig(f'{plot_title}_nolegend.png', dpi=200, bbox_inches = 'tight')

fig1.legend(fancybox=False, edgecolor='k', framealpha=0.8, shadow=True)
fig1.savefig(f'{plot_title}_withlegend_small.png', bbox_inches = 'tight')
fig1.savefig(f'{plot_title}_withlegend.png', dpi=200, bbox_inches = 'tight')
