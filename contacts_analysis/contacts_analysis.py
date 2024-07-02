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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import contacts
import argparse
import logging
#%matplotlib inline
import warnings
import sys
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

# define universe
u = mda.Universe(topol, traj)

# create analyser
def contacts_within_cutoff(u, group_a, group_b, radius):
    timeseries = []
    for ts in u.trajectory[frame_start:frame_stop]: # these values are FRAMES, not TIMES. since t between frames is 2 ps, this is 90000 to 100000 ps
        # calculate distances between group_a and group_b
        dist = contacts.distance_array(group_a.positions, group_b.positions)
        # determine which distances <= radius
        n_contacts = contacts.contact_matrix(dist, radius).sum()
        timeseries.append([ts.frame, n_contacts])
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

# print(results)


# iterate over selections
for i in sel:
    
    # define selection for an iteration
    group_b = u.select_atoms(f'{i}')
    #group_b_array = np.array(group_b)
    
    # define radius depending on the ref/sel pair
    #### for future version of this code - consider creating a separate file where these distances are defined and just "importing it" here as the code runs ####
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

##### to be removed
# stack the time and contacts arrays, then create a transpose
# contacts_vs_time  = np.transpose(np.stack((results, contacts)))
#####

# Axis labels (this bit is a bit clunky)
# Create list with just first axis label, then extend it to include all the selections
contacts_labels = ['Time (ns)']
contacts_labels.extend(sel)

# Create a dataframe with the data and datalabels we just defined
contacts_vs_time_df = pd.DataFrame(transposed_results, columns = contacts_labels)
contacts_vs_time_df.plot(x='Time (ns)')
plt.ylabel('Number of contacts')


# SAVING
#define plot title
num_sel = len(sel) #for filename - since can't add a list to a filename, instead we will just indicate the number of selections in the filename
plot_title = f'contacts_ref_{ref}_{num_sel}sel_{frame_start}to{frame_stop}'
plot_title = plot_title.replace(' ','_') # replace whitespaces with underscores
plt.savefig(f'{plot_title}.png', bbox_inches='tight') # save figure

# #%%

# # OLD VERSION
# results = []

# # iterate over selections
# for i in sel:
    
#     # define selection for an iteration
#     group_b = u.select_atoms(f'{i}')
#     #group_b_array = np.array(group_b)
    
#     if ref == 'type Uo1':
#         if i == 'name OW*':
#             radius = 2.46
#         elif i == 'type OG2D2':
#             radius = 2.34
#         elif i == 'type Oc*':
#             radius = 2.38
#         elif i == 'type OB*':
#             radius = 2.46
#     elif ref == 'type No1':
#         if i == 'name OW*':
#             radius = 2.54
#         elif i == 'type OG2D2':
#             radius = 2.42
#         elif i == 'type Oc*':
#             radius = 2.46
#         elif i == 'type OB*':
#             radius = 2.46

#     #run the analysis
#     run = contacts_within_cutoff(u, group_a, group_b, radius)

#     results.append(run)

#     # create array of results
#     cont_vs_time = np.transpose(run)

#     # extract and convert # frames to time (ns) - divide by conversion factor (500 frames ns^-1)
#     t_ns = cont_vs_time[0] / 500

#     #extract contacts, redefine as coordinating oxygens 

#     coord_oxy = cont_vs_time[1]

#     contacts_vs_time  = np.transpose(np.stack((t_ns, coord_oxy)))

#     #make dataframe
#     contacts_vs_time_df = pd.DataFrame(contacts_vs_time, columns = ['Time (ns)', '# contacts'])
#     contacts_vs_time_df.to_csv(f"{ref}_{i}_contacts.csv")

    
#     # plot the pd dataframe
#     contacts_vs_time_df.plot(x='Time (ns)')
#     plt.ylabel('Contacts')

    # # SAVING
    # #define plot title
    # plot_title = f'contacts_ref_{ref}_sel{i}_{frame_start}to{frame_stop}'
    # #replace whitespaces with underscores
    # plot_title = plot_title.replace(' ','_')
    # #save figure
    # plt.savefig(f'{plot_title}.png', bbox_inches='tight')






# #%%
# #one x, multiple y
# fig, ax = plt.subplots()


# #define axes - ax.plot(x, y, line colour, label to show up in legend)
# ax.plot(distance, dens2, c = 'k', label = 'MAL', linestyle='dashed') 
# ax.plot(distance, dens3, c = 'b', label = 'Na', linestyle='dashed') 
# #ax.plot(distance, dens4, c = 'k', label = 'CO3', linestyle='solid') 
# ax.plot(distance, dens1, c = 'm', label = 'UO2', linewidth='3') 

# #b = blue, g = green,r = red, c = cyan, m = magenta, y=yellow, k = black, w=white

# #axis labels
# ax.set_xlabel('distance (nm)')
# ax.set_ylabel('number density')

# #axis limits
# plt.xlim(-4.5,4.5)
# plt.ylim(0,4)

# #add a legend and title
# leg = ax.legend(loc = 'upper right')

# plt.title("6 UO2, 12 MAL")

# #set plot size (how to scale axes)
# plt.rcParams['figure.figsize'] = [10,3]
# plt.show()
