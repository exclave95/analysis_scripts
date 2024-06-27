# %% [markdown]
# CONTACTS BASED ON MDANALYSIS: https://userguide.mdanalysis.org/stable/examples/analysis/distances_and_contacts/distances_between_atomgroups.html

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import contacts
import argparse
#%matplotlib inline
import warnings
# suppress some MDAnalysis warnings when writing PDB files
warnings.filterwarnings('ignore')

# parse user input arguments
parser = argparse.ArgumentParser(description="Specify analysis parameters")
parser.add_argument('-t', help='trajectory file')
parser.add_argument('-s', default='topol.tpr', help='topology file')
parser.add_argument('-r', help='size of sampling radius to define a contact')
parser.add_argument('-ref', help='reference species')
parser.add_argument('-sel', help='selection species. NOTE: string needs to be in quotation marks, separate selections with commas')
# parser.add_argument('-ts', default=2, help='timestep (in ps) BETWEEN FRAMES')
parser.add_argument('-start', default=0, help='initial frame to read')
parser.add_argument('-stop', default=-1, help='final frame to read')
args = vars(parser.parse_args())

# convert user inputs into variables to use later
radius = float(args["r"])
ref = args['ref']
sel = args['sel'].split(', ')
traj = args['t']
topol = args['s']
#ts = int(args['ts'])
frame_start = int(args['start'])
frame_stop = int(args['stop'])


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


# iterate over selections
for i in sel:
    
    # define selection for an iteration
    group_b = u.select_atoms(f'{i}')
    #group_b_array = np.array(group_b)
    
    if ref == 'type Uo1':
        if i == 'name OW*':
            radius = 2.46
        elif i == 'type OG2D2':
            radius = 2.34
        elif i == 'type Oc*':
            radius = 2.38
        elif i == 'type OB*':
            radius == 2.46
    elif ref == 'type No1':
        if i == 'name OW*':
            radius = 2.54
        elif i == 'type OG2D2':
            radius = 2.42
        elif i == 'type Oc*':
            radius = 2.46
        elif i == 'type OB*':
            radius == 2.46

    #run the analysis
    run = contacts_within_cutoff(u, group_a, group_b, radius)

    # create array of results
    cont_vs_time = np.transpose(run)

    # extract and convert # frames to time (ns) - divide by conversion factor (500 frames ns^-1)
    t_ns = cont_vs_time[0] / 500

    #extract contacts, redefine as coordinating oxygens 

    coord_oxy = cont_vs_time[1]

    contacts_vs_time  = np.transpose(np.stack((t_ns, coord_oxy)))

    #make dataframe
    contacts_vs_time_df = pd.DataFrame(contacts_vs_time, columns = ['Time (ns)', '# contacts'])
    contacts_vs_time_df.to_csv(f"{ref}_{i}_contacts.csv")

    
    # plot the pd dataframe
    contacts_vs_time_df.plot(x='Time (ns)')
    plt.ylabel('Contacts')

    # SAVING
    #define plot title
    plot_title = f'contacts_ref_{ref}_sel{i}_{frame_start}to{frame_stop}'
    #replace whitespaces with underscores
    plot_title = plot_title.replace(' ','_')
    #save figure
    plt.savefig(f'{plot_title}.png', bbox_inches='tight')


