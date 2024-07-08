### contacts analysis for multiple selections

# %%
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
parser.add_argument('-stdev', default = 1, help='number of stdevs to plot for uncertainty')
parser.add_argument('-radius', help='pick uniform radius for all selections')
parser.add_argument('-aw', default = 100, help="averaging window for plotting. default 100")
parser.add_argument('-csv', default = 'yes', choices=['yes','no'], help="save results to csv? default yes")
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
    plot_title = f'contacts_ref_{ref}_{num_sel}sel_{frame_start}to{frame_stop}_{stdev}stdev_{aw}aw_{radius}A'
else:
    plot_title = f'contacts_ref_{ref}_{num_sel}sel_{frame_start}to{frame_stop}_{stdev}stdev_{aw}aw_indivradii'
plot_title = plot_title.replace(' ','_') # replace whitespaces with underscores


# define universe
u = mda.Universe(topol, traj)

# %%
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

### ADD LINESTYLE CYCLER ###

# %%
for i in sel:
    dataset = transposed_results_df[i]
    dataset_rollavg = dataset.rolling(aw).mean()
    dataset_mstd = dataset.rolling(aw).std()
    dataset_rollavg.plot()
    #### alt plotting method
    #import seaborn as sns
    #sns.lineplot(data = dataset_rollavg, label = f'{i}')
    plt.fill_between(dataset.index, dataset_rollavg - stdev * dataset_mstd, dataset_rollavg + stdev * dataset_mstd, alpha=0.4)

plt.xlabel('Frame')
plt.legend()
plt.ylabel('contacts')

# num_sel = len(sel) #for filename - since can't add a list to a filename, instead we will just indicate the number of selections in the filename
# plot_title = f'contacts_ref_{ref}_{num_sel}sel_{frame_start}to{frame_stop}_{stdev}stdev_{aw}aw'
# plot_title = plot_title.replace(' ','_') # replace whitespaces with underscores
plt.savefig(f'{plot_title}.png', bbox_inches='tight') # save figure


# %%
# plt.figure()
# plt.plot(contacts_only_df)


