#! /usr/bin/python
#
##### MULTIPLE CONTACT ANALYSIS AND PLOTTING ##### 
#              Written by Jakub LICKO, MChem       
# Based on and adapted from the instructions in official MDAnalysis documentation: 
# https://userguide.mdanalysis.org/stable/examples/analysis/distances_and_contacts/distances_between_atomgroups.html
#
#  INSTRUCTIONS
#  Example command line input:
#       python contacts_analysis.py -t trajout.xtc -s topol.tpr -ref "name Uo1" -sel "name OW*, name OG2D2" -start 45000 -stop -1
#  Flags:
#       -t : centred trajectory file
#       -s : topology file (tpr for GROMACS)
#       -ref : reference atom
#       -sel : selection of species, in quotation marks, comma separated selections (e.g. "resname SOL, resname Na")
#       -start : first trajectory frame to analyse
#       -stop : final trajectory frame to analyse (default -1, i.e. last frame)
# 
#  PREREQUISITES
#  Installed python libraries: 
#       numpy
#       pandas
#       matplotlib.pyplot
#       MDAnalysis
#       argparse
#       sys
#       logging

# %%
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import contacts
import argparse
import scienceplots
import logging
import warnings
import sys
from cycler import cycler
import itertools

# suppress some MDAnalysis warnings when writing PDB files
warnings.filterwarnings('ignore')

# parse user input arguments
parser = argparse.ArgumentParser(description="Specify analysis parameters")
parser.add_argument('-t', help='trajectory file')
parser.add_argument('-s', default='topol.tpr', help='topology file')
# parser.add_argument('-r', help='size of sampling radius to define a contact') - commented since I define my radii in the code itself
parser.add_argument('-ref', help='reference species')
parser.add_argument('-sel', help='selection species. NOTE: string needs to be in quotation marks, separate selections with comma + space, e.g. resname A, resname B')
# parser.add_argument('-ts', default=2, help='timestep (in ps) BETWEEN FRAMES')
parser.add_argument('-start', default=0, help='initial frame to read')
parser.add_argument('-stop', default=-1, help='final frame to read')
parser.add_argument('-stdev', default = 0, help='number of stdevs to plot for uncertainty')
parser.add_argument('-radius', help='pick uniform radius for all selections')
parser.add_argument('-aw', default = 100, help="averaging window for plotting. default 100")
parser.add_argument('-csv', default = 'yes', choices=['yes','no'], help="save results to csv? default yes")
parser.add_argument('-m', default = 'median', choices=['median','average'], help="dataset manipulation")

args = vars(parser.parse_args())

# convert user inputs into variables to use later
# radius = float(args["r"])
ref = args['ref']
sel = args['sel'].split(', ')
traj = args['t']
topol = args['s']
#ts = int(args['ts'])
frame_start = int(args['start'])
frame_stop = int(args['stop'])
stdev = int(args['stdev'])
radius = float(args['radius'])
aw = int(args['aw'])
csv = args['csv']
m = args['m']

# logging 
logname = "contacts_analysis.log"
logger = logging.getLogger("contacts_analysis")
fh = logging.FileHandler(logname)
ch = logging.StreamHandler()
logger.addHandler(fh)
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)
logger.info(" ".join(sys.argv))
logger.info("")

# plot title for later
# num_sel = len(sel) #for filename - since can't add a list to a filename, instead we will just indicate the number of selections in the filename
if radius:
    plot_title = f'contacts_ref_{ref}_{frame_start}to{frame_stop}_{m}_{aw}aw_{radius}A'
# else:
#     plot_title = f'contacts_ref_{ref}_{frame_start}to{frame_stop}_{test}_{aw}aw_indivradii'
plot_title = plot_title.replace(' ','_') # replace whitespaces with underscores


# define universe
u = mda.Universe(topol, traj)

# %%
# create analyser
def contacts_within_cutoff(u, group_a, group_b, radius):
    results_timeseries = []
    for ts in u.trajectory[frame_start:frame_stop]: # these values are FRAMES, not TIMES
        # calculate distances between group_a and group_b
        dist = contacts.distance_array(group_a.positions, group_b.positions)
        # determine which distances <= radius
        n_contacts = contacts.contact_matrix(dist, radius).sum()
        # timeseries.append([ts.frame, n_contacts])

        # Create ns timeseries
        t_ps = ts.frame * 2
        t_ns = t_ps / 1000
        results_timeseries.append([t_ns, n_contacts])
	#time_timeseries = results_timeseries[0]
    return np.array(results_timeseries)

# define reference. this doesn't change
group_a = u.select_atoms(f'{ref}')

# initialise plotting and set plot styles
plt.style.use(['science','notebook','grid','no-latex'])
fig, ax = plt.subplots()
# colour iterator for looped plotting
colours = itertools.cycle(("red", "green", "blue", "orange"))
#plotting and median/mean loop
for i in sel:
    group_b = u.select_atoms(f'{i}')
    #run the analysis
    run = contacts_within_cutoff(u, group_a, group_b, radius) 
    run = np.transpose(run)
    # append contacts array to previously defined time array
   # results = np.vstack((results, run[1]))
    # rinse and repeat
    
    #print(run)
    #print(np.shape(run))
    time_timeseries = run[0]
    contacts_timeseries = run[1]    

    # print('time', time_timeseries)
    # print('contacts', contacts_timeseries)
    if m == "median":
        ######### calculate rolling median
        # first convert to df
        contacts_df = pd.DataFrame(contacts_timeseries)
        # rolling median df
        dataset_rollmedian_df = contacts_df.rolling(aw).median()
        # convert back to numpy array
        dataset_rollmedian_array = dataset_rollmedian_df.to_numpy()

        colour = next(colours)
        plt.plot(time_timeseries, dataset_rollmedian_array, c=colour, label = f'{i}')

    else:
        ######### calculate rolling average
        # first convert to df
        contacts_df = pd.DataFrame(contacts_timeseries)
        # rolling average df
        dataset_rollaverage_df = contacts_df.rolling(aw).mean()
        # convert back to numpy array
        dataset_rollaverage_array = dataset_rollaverage_df.to_numpy()

        colour = next(colours)
        plt.plot(time_timeseries, dataset_rollaverage_array, c=colour, label = f'{i}')
    
    # csv file - replace whitespaces and asterisks for better filenaming practice
    csv_filename = f'{i}'
    csv_filename = csv_filename.replace(' ','_')
    csv_filename = csv_filename.replace('*','all')
    np.savetxt(f'{csv_filename}.csv', contacts_timeseries, delimiter = ',')
   

plot_title = f'contacts_ref_{ref}_{frame_start}to{frame_stop}_{m}_{aw}aw'
plot_title = plot_title.replace(' ','_') # replace whitespaces with underscores

# # plt.grid() redundant since i specified 'grid' in the matplotlib.style.use function call
# plt.rc('axes', prop_cycle = default_cycler)
ax.set_xlabel('Time (ns)')
# ax.legend()
ax.set_ylabel(f'{m} no. of contacts')
ax.set_ylim(-1, 33)

#save figure - multiple options for presentations, thesis, publications, etc
plt.savefig(f'{plot_title}_small.png', bbox_inches='tight')
plt.savefig(f'{plot_title}_nolegend.png', dpi=200, bbox_inches = 'tight')

ax.legend()
plt.savefig(f'{plot_title}_withlegend_small.png', bbox_inches = 'tight')
plt.savefig(f'{plot_title}_withlegend.png', dpi=200, bbox_inches = 'tight')
