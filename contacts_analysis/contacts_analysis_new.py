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
parser.add_argument('-stdev', default = 1, help='number of stdevs to plot for uncertainty')
parser.add_argument('-radius', help='pick uniform radius for all selections')
parser.add_argument('-aw', default = 100, help="averaging window for plotting. default 100")
parser.add_argument('-csv', default = 'yes', choices=['yes','no'], help="save results to csv? default yes")
parser.add_argument('-test', choices=['median','average'], help="save results to csv? default yes")

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
test = args['test']

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
num_sel = len(sel) #for filename - since can't add a list to a filename, instead we will just indicate the number of selections in the filename
if radius:
    plot_title = f'contacts_ref_{ref}_{num_sel}sel_{frame_start}to{frame_stop}_{test}_{stdev}stdev_{aw}aw_{radius}A'
else:
    plot_title = f'contacts_ref_{ref}_{num_sel}sel_{frame_start}to{frame_stop}_{test}_{stdev}stdev_{aw}aw_indivradii'
plot_title = plot_title.replace(' ','_') # replace whitespaces with underscores


# define universe
u = mda.Universe(topol, traj)

# %%
# create analyser
def contacts_within_cutoff(u, group_a, group_b, radius):
    timeseries = []
    for ts in u.trajectory[frame_start:frame_stop]: # these values are FRAMES, not TIMES
        # calculate distances between group_a and group_b
        dist = contacts.distance_array(group_a.positions, group_b.positions)
        # determine which distances <= radius
        n_contacts = contacts.contact_matrix(dist, radius).sum()
        timeseries.append([ts.frame, n_contacts])

# Idea
# t_ps = ts.frame * 2
# t_ns = t_ps / 1000
# timeseries.append([t_ns, n_contacts])

    return np.array(timeseries)

# define reference. this doesn't change
group_a = u.select_atoms(f'{ref}')

# NEW VERSION
# create separate array for frames only
frames = []
for ts in u.trajectory[frame_start:frame_stop]:
    frames.append([ts.frame])
#convert to array
results = np.array(frames)
#convert to ns
results = np.transpose(results / 500)

# %%
# iterate over selections
for i in sel:
    
    # define selection for an iteration
    group_b = u.select_atoms(f'{i}')
    #group_b_array = np.array(group_b)
    
    # define radius depending on the ref/sel pair
    #### for future version of this code - consider creating a separate file where these distances are defined and just "importing it" here as the code runs ####
    # if a radius was specified (i.e. a uniform sphere), do nothing
    if radius:
        pass
    # otherwise, use the following definitions
    else:
        if ref == 'type Uo1' or ref == 'name Uo1':
            if i == 'name OW*':
                radius = 2.46
            elif i == 'type OG2D2':
                radius = 2.34
            elif i == 'name Oc*':
                radius = 2.38
            elif i == 'name OB*':
                radius = 2.46
        elif ref == 'type No1' or ref == 'name No1':
            if i == 'name OW*':
                radius = 2.54
            elif i == 'type OG2D2':
                radius = 2.42
            elif i == 'name Oc*':
                radius = 2.46
            elif i == 'name OB*':
                radius = 2.46
        
    #run the analysis
    run = contacts_within_cutoff(u, group_a, group_b, radius) 
    run = np.transpose(run)
    # append contacts array to previously defined time array
    results = np.vstack((results, run[1]))
    # rinse and repeat

# process the data and plot it

# transpose results array
transposed_results = np.transpose(results)
# print(transposed_results)

# %%
transposed_results_df = pd.DataFrame(transposed_results)
print(transposed_results_df)

# %%
contacts_labels = ['Time (ns)']
contacts_labels.extend(sel)

transposed_results_df.columns = contacts_labels
transposed_results_df.iloc[:,0]

if csv == 'yes':
    transposed_results_df.to_csv(f"{plot_title}.csv")

# %%
# print(transposed_results_df.iloc[0:,0:1])

# contacts_only_df = transposed_results_df.iloc[:, 1:]
# contacts_only_df

### LINESTYLE CYCLER ###
default_cycler = (cycler(color=['r', 'g', 'b', 'orange']) +
                  cycler(linestyle=['-', '--', ':', '-.']))

# %%
plt.style.use(['science','notebook','grid','no-latex'])

fig, ax = plt.subplots()


#%% analysis 
# option: rolling AVERAGE
if test == 'average':
    for i in sel:
        dataset = transposed_results_df[i]
        dataset_rollavg = dataset.rolling(aw).mean()
        dataset_mstd = dataset.rolling(aw).std()
        dataset_rollavg.plot()
        #### alt plotting method
        #import seaborn as sns
        #sns.lineplot(data = dataset_rollavg, label = f'{i}')
        plt.fill_between(dataset.index, dataset_rollavg - stdev * dataset_mstd, dataset_rollavg + stdev * dataset_mstd, alpha=0.4)

    # num_sel = len(sel) #for filename - since can't add a list to a filename, instead we will just indicate the number of selections in the filename
    # plot_title = f'contacts_ref_{ref}_{num_sel}sel_{frame_start}to{frame_stop}_{stdev}stdev_{aw}aw'
    # plot_title = plot_title.replace(' ','_') # replace whitespaces with underscores

# option: rolling MEDIAN
elif test == "median":
    for i in sel:
        dataset = transposed_results_df[i]
        dataset_rollmedian = dataset.rolling(aw).median()
        dataset_rollmedian.plot()
        #### alt plotting method
        #import seaborn as sns
        #sns.lineplot(data = dataset_rollavg, label = f'{i}')

    # num_sel = len(sel) #for filename - since can't add a list to a filename, instead we will just indicate the number of selections in the filename
    # plot_title = f'contacts_ref_{ref}_{num_sel}sel_{frame_start}to{frame_stop}_{stdev}stdev_{aw}aw'
    # plot_title = plot_title.replace(' ','_') # replace whitespaces with underscores

# plt.grid() redundant since i specified 'grid' in the matplotlib.style.use function call
plt.rc('axes', prop_cycle = default_cycler)
ax.set_xlabel('Frame')
# ax.legend()
ax.set_ylabel('contacts')

#save figure - multiple options for presentations, thesis, publications, etc
plt.savefig(f'{plot_title}_small.png', bbox_inches='tight')
plt.savefig(f'{plot_title}_nolegend.png', dpi=200, bbox_inches = 'tight')

ax.legend()
plt.savefig(f'{plot_title}_withlegend_small.png', bbox_inches = 'tight')
plt.savefig(f'{plot_title}_withlegend.png', dpi=200, bbox_inches = 'tight')
