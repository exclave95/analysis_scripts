#! /usr/bin/python
#
##### SURVIVAL PROBABILITY PLOT FOR MOLECULAR DYNAMICS SIMULATIONS ##### 
#              Written by Jakub LICKO, MChem       
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
#
#  PREREQUISITES
#   (1) Installed python libraries: 
#       numpy
#       matplotlib.pyplot
#       MDAnalysis
#       argparse
#       sys
#       logging
#       time
#       cycler

import numpy as np 
# import pandas as pd
# import matplotlib as mpl
import matplotlib.pyplot as plt
import MDAnalysis as mda
import argparse
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
parser.add_argument('-taumax', default=20, help='number of frames to compute SP for - note that this may be converted to time with the -ts flag')

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
# csv = args['csv']


# logging 
logname = "surfdensmap.log"
logger = logging.getLogger("surfdensmap")
fh = logging.FileHandler(logname)
ch = logging.StreamHandler()
logger.addHandler(fh)
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)
logger.info(" ".join(sys.argv))
logger.info("")
        
# start time
start_time = time.time()

# define universe
u = mda.Universe(topol, traj)


#%%
# MAIN CELL #

from cycler import cycler
default_cycler = (cycler(color=['r', 'g', 'b', 'orange']) +
                  cycler(linestyle=['-', '--', ':', '-.']))

for i in sel:
    select = f"{i} and sphzone {radius} {ref}"
    sp = SP(u, select, verbose=True)
    sp.run(start=frame_start, stop=frame_stop, tau_max=taumax)
    tau_timeseries = sp.tau_timeseries

    # START OF MODIFICATION
    # original function takes no account for time between frames
    # so while tau is defined in terms of time in original documentation, it isn't necessarily the case
    # hence, I introduce a multiplier here - the difference between frames is 2 ps, i.e. 0.002 ns (50 000 frames == 100 000 ps == 100 ns)
    # multiplier = 2
    time_timeseries = [x * ts for x in tau_timeseries]
    # END OF MODIFICATION

    sp_timeseries = sp.sp_timeseries
    # print in console
    for tau, sp in zip(tau_timeseries, sp_timeseries):
        print("{time} {sp}".format(time=tau, sp=sp))
    plt.plot(time_timeseries, sp_timeseries, label = i)

plt.grid()
plt.rc('axes', prop_cycle = default_cycler)
plt.xlabel('Time')
plt.ylabel('SP')
plt.legend()
plt.title('survival probability')

#%%
# SAVING     
# define plot title
plot_title = f'{i}_survprob_frame{frame_start}to{frame_stop}_tau{taumax}_ref{ref}'
#replace whitespaces with underscores
plot_title = plot_title.replace(' ','_')
#save figure
plt.savefig(f'{plot_title}.png', bbox_inches='tight')