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
#       -radius : radius around ref to compute SP around
#       -sel : selection of species, in quotation marks, comma separated selections (e.g. "resname SOL, resname Na")
#       -start : first trajectory frame to analyse
#       -stop : final trajectory frame to analyse (default -1, i.e. last frame)
#       -csv : save info to csv file? 
#       -taumax : number of frames to compute SP for
#
#
#   Required installed python libraries: 
#       numpy
#       matplotlib.pyplot
#       itertools
#       scipy
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
import sys
import logging
from cycler import cycler
from time import time

# parse user input arguments
parser = argparse.ArgumentParser(description="Specify analysis options")
parser.add_argument('-t', help='trajectory file')
parser.add_argument('-s', default='topol.tpr', help='topology file')
parser.add_argument('-ref', help='which atom to compute a survival probability around')
parser.add_argument('-radius', help='radius around selection to compute survival probability around')
parser.add_argument('-sel', help='species to sample. NOTE: string needs to be in quotation marks, separate selections with commas')
parser.add_argument('-ts', default=2, help='timestep (in ps) BETWEEN FRAMES')
parser.add_argument('-start', default=0, help='initial frame to read')
parser.add_argument('-stop', default=-1, help='final frame to read')
parser.add_argument('-csv', choices=['yes','no'], default = 'yes', help='Save SP of ligands to csv files? Options: yes (default), no')
parser.add_argument('-taumax', default=20, help='number of frames to compute SP for')

# parser.add_argument('-csv', choices=['yes','no'], default = 'yes', help='Save positions of selections and substitution sites to csv files? Options: yes (default), no')
args = vars(parser.parse_args())

# convert user inputs into variables to use later
ref = args['ref']
sel = args['sel'].split(', ')
traj = args['t']
topol = args['s']
radius = float(args['radius'])
ts = int(args['ts'])
frame_start = int(args['start'])
frame_stop = int(args['stop'])
taumax = int(args['taumax'])
# plot_type = args['plot'])
csv = args['csv']


# logging 
logname = "surv_prob.log"
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
u = mda.Universe(topol, traj)

# find current directory
cwd = os.getcwd()

# define curve fitting function
def surv_prob_curve_fit():
    # data prep - specifies how the SP timeseries data will be used by the function
    x = time_timeseries
    y = sp_timeseries
    
    # fitting (x,y) data to the exponential decay curve: 
    # y = a * exp(-k * x) + c

    # parameters to be fitted: 
    # a = pre-exponential factor 
        # needed along with c to ensure that y is ~= 1 when x = 0
    # k = decay coefficient 
        # this is the key parameter for comparison between selected species
    # c = constant, serves as the horizontal asymptote - 
        # needed since decay curves are unlikely to approach 0, 
        # and needed along with a to ensure y is ~= 1 when x = 0
    
    # give parameters global scope (so the code can recognise them when the function is called)
    global popt, pcov, perr, a, k, c, x_fitted, y_fitted

    # define optimization parameters and their covariance coefficients
    popt, pcov = curve_fit(lambda t, a, k, c: a * (np.exp(-k * t)) + c, x, y)

    # define a, b and c 
    a = popt[0]
    k = popt[1]
    c = popt[2]

    # define fitted x and y
    x_fitted = np.linspace(np.min(x), np.max(x), 100)
    y_fitted = a * np.exp(-k * x_fitted) + c

    # create array of the error values of each fitted parameter
    # this is done by identifying the DIAGONAL values of the covariance matrix (popt), and then calculating their square root
    perr = np.sqrt(np.diag(pcov))
    
    # LEGACY plotting code, kept from original curve_fit tutorial (link: HERE)
    # ax = plt.axes()
    # ax.scatter(x, y, label='Raw data')
    # ax.plot(x_fitted, y_fitted, 'k', label='Fitted curve')
    # ax.set_title(r'Using curve_fit() to fit an exponential function')
    # ax.set_ylabel('y-Values')
    # ax.set_ylim(0, 1)
    # ax.set_xlabel('x-Values')
    # ax.legend()


# create results file
with open("surv_prob_results.txt", "w") as file:
    file.write('Survival Probability')
    file.write('\n--------------------')
    file.write(f"\nCalculated in directory: {cwd}")
    file.write(f'\nReference: {ref}\nFull selection: {sel}')
    file.write(f'\nradius: {radius} Angstroms\nframes: {frame_start} to {frame_stop}\ntau: {taumax}')

#%%
# MAIN CELL #
# plot formatting cycler - line colours and line styles
# default_cycler = (cycler(color=['r', 'g', 'b', 'orange']) +
                #   cycler(linestyle=['-', '--', ':', '-.']))

# define scatter plot colours and markers
marker = itertools.cycle(('o', '+', 'x', '*'))
colours = itertools.cycle(("red", "green", "blue", "orange"))

fig, ax = plt.subplots()

# calculation loop
for i in sel:
    # select reference and selection pair and calculate SP for it
    select = f"{i} and sphzone {radius} {ref}"
    sp = SP(u, select, verbose=True)
    sp.run(start=frame_start, stop=frame_stop, tau_max=taumax)
    tau_timeseries = sp.tau_timeseries

    # MODIFICATION 1 - TIMESTEP MULTIPLIER
    # original function takes no account for time between frames (i.e. the TIMESTEP)
    # so while tau is defined in terms of time in the original documentation, it isn't always necessarily the case
    # hence, I introduce a TIMESTEP (ts) here - the difference between frames is 2 ps, i.e. 0.002 ns (50 000 frames == 100 000 ps == 100 ns)
    # the value of the TIMESTEP is 2 by default, but it can be defined by the user with the -ts flag
    time_timeseries = [x * ts for x in tau_timeseries]
    # END OF MODIFICATION 1

    # define the surv prob array
    sp_timeseries = sp.sp_timeseries
    
    # print in console (optional, retained from the original documentation)
    for tau, sp in zip(tau_timeseries, sp_timeseries):
        print("{time} {sp}".format(time=tau, sp=sp))

    # MODIFICATION 2 - saving to CSV file 
    if csv == 'yes':
        print(f'Saving {i} values into csv file')
        surv_prob_data = time_timeseries, sp_timeseries
        surv_prob_data = np.transpose(surv_prob_data)
        
        # define filename, replace whitespaces with underscores and asterisks with 'all' 
        csv_filename = f'{i}_surv_prob'
        csv_filename = csv_filename.replace(' ','_')
        csv_filename = csv_filename.replace('*','all')

        # save as a csv file 
        np.savetxt(f'{csv_filename}.csv', surv_prob_data, delimiter = ',', header=f'SP timeseries of sel {i} within {radius} A of ref {ref}\n{cwd}')  

    else:
        print('Survival probability not being saved')   
    # END OF MODIFICATION 2

    # MODIFICATION 3 - FITTING THE SP DATA TO A CURVE AND SAVING CURVE PARAMETERS TO TEXT FILE
    # the function was defined earlier in the code for clarity, and is simply called here
    surv_prob_curve_fit()
    
    # append results to txt file
    with open('surv_prob_results.txt', 'a') as file:
        file.write('\n--------------------')
        file.write(f'\n{i}')
        file.write(f'\na = {a}\nk = {k}\nc = {c}')
        file.write(f'\n\nCovariance matrix:')
        file.write(f'\n{pcov}')
        file.write(f'\n\nParameter error values:')
        file.write(f'\na error = {perr[0]}\nk error = {perr[1]}\nc error = {perr[2]}')
        
    # END OF MODIFICATION 3

    # plotting
    colour = next(colours) #ensures same colour for both points and curve
    plt.scatter(time_timeseries, sp_timeseries, label = i, c=colour, marker=next(marker))
    plt.plot(x_fitted, y_fitted, c=colour)


# ORDER IS IMPORTANT - plt.rc(...) must be first, THEN plt.grid()
# plt.rc('axes', prop_cycle = default_cycler)
plt.grid()

ax.set_xlabel('Time (ps)')
ax.set_ylabel('SP')
# ax.legend()
plt.title(f'SP - {radius} A of {ref}')

#%%
# PLOT GENERATION AND SAVING     

# define plot title
plot_title = f'survprob_frame{frame_start}to{frame_stop}_tau{taumax}_ref{ref}'

#replace whitespaces with underscores and asterisks with
plot_title = plot_title.replace(' ','_')

#save figure
plt.savefig(f'{plot_title}.png', bbox_inches='tight')


