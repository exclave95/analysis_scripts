#! /usr/bin/python
#
##### SURFACE DENSITY MAP GENERATOR FOR MOLECULAR DYNAMICS SIMULATIONS ##### 
#              Written by Jakub LICKO, MChem       
#
#  INSTRUCTIONS
#  Example command line input:
#       python surf_dens.py -t trajout.xtc -s topol.tpr -z0 0 -dz 1.5 -sel "resname UO2, resname MAL" -start 45000 -stop -1 -plot heatmap 
#  Flags:
#       -t : centred trajectory file
#       -s : topology file (tpr for GROMACS)
#       -z0 : distance from clay surface to consider, default 0
#       -dz : thickness of sampling layer (in Angstroms)
#       -sel : selection of species, in quotation marks, comma separated selections (e.g. "resname SOL, resname Na")
#       -start : first trajectory frame to analyse
#       -stop : final trajectory frame to analyse (default -1, i.e. last frame)
#       -plot : type of plot to generate. Options: heatmap (default, recommended), scatter
#       -side : which 'face' to sample. Options: top (default), bottom, both. 
#               Note 1: 'top' and 'bottom' refer to the z-coordinates of exposed clay surfaces.
#               Top = maximum z-coords, bottom = minimum z-coords
#               Note 2: 'both' option only works when both top and bottom layers are fixed in place, e.g. if position restraints are applied on all clay layers
#
#  PREREQUISITES
#   (1) Installed python libraries: 
#       numpy
#       pandas
#       matplotlib.pyplot
#       MDAnalysis
#       argparse
#       sys
#       logging
#   (2) Trajectory preprocessing: clay surface being sampled needs to remain fixed
#       (a) Select atom in clay surface to be sampled (e.g. atom 3521)
#       (b) Create a Gromacs index containing chosen atom
#       e.g.    gmx_mpi make_ndx -f confout_PROD.gro -o surf_dens.ndx
#       (c) Center trajectory using the index, keeping the surface centred (-center) and all atoms in simulation box (-pbc atom)       
#       e.g.    gmx_mpi trjconv -f ../traj.trr -s ../topol.tpr -o ${d}_trajout_center.xtc -n traj_center.ndx -center -pbc atom

import numpy as np 
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import MDAnalysis as mda
import argparse
import sys
import logging
from time import time
import matplotlib.colors as mplcolors
# scipy.stats as st imported later in code

# parse user input arguments
parser = argparse.ArgumentParser(description="Specify analysis options")
parser.add_argument('-t', help='trajectory file')
parser.add_argument('-s', default='topol.tpr', help='topology file')
parser.add_argument('-z0', default='0', help='distance (Angstrom) from clay surface. default = 0, i.e. clay surface')
parser.add_argument('-dz', help='thickness of sampling layer (Angstrom)')
parser.add_argument('-sel', help='species to sample. NOTE: string needs to be in quotation marks, separate selections with commas')
# parser.add_argument('-ts', default=2, help='timestep (in ps) BETWEEN FRAMES')
parser.add_argument('-start', default=0, help='initial frame to read')
parser.add_argument('-stop', default=-1, help='final frame to read')
parser.add_argument('-plot', choices=['contour','contourf','heatmap','scatter'], default='contour', help='type of plot to generate. Options: heatmap (default), scatter')
parser.add_argument('-side', choices=['top','bottom'], default='top', help='which exposed clay layer to use for analysis. default: top, i.e. the face with the highest z coordinates')
parser.add_argument('-csv', choices=['yes','no'], default = 'yes', help='Save positions of selections and substitution sites to csv files? Options: yes (default), no')
args = vars(parser.parse_args())

# convert user inputs into variables to use later
z0 = float(args["z0"])
dz = float(args["dz"])
sel = args['sel'].split(', ')
traj = args['t']
topol = args['s']
#ts = int(args['ts'])
frame_start = int(args['start'])
frame_stop = int(args['stop'])
plot_type = args['plot']
side = args['side']
csv = args['csv']


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
        

def setup():
    # make them global so they can be accessed in isomorphous_substitutions() and analysis() functions
    global minX, maxX, minY, maxY, minZ, maxZ, clay_min_z, clay_max_z, u

    # define universe
    u = mda.Universe(topol, traj)

    # dimensions of simulation box (Sanity check):
    box_dim = u.dimensions
    minX, maxX = 0, box_dim[0]
    minY, maxY = 0, box_dim[1]
    minZ, maxZ = 0, box_dim[2]

    # %%
    # select clay atoms
    clay = u.select_atoms('resname UC*')

    # create array containing clay positions, with x y and z being in separate rows
    # i.e. transform FROM N atoms with 3 coordinates TO 3 coordinate sets with N entries each
    clay_positions = np.transpose(clay.positions)

    # identify max and min z coordinates of all clay atoms (only max is needed for SDM generation)
    clay_min_z = np.min(clay_positions[2])
    clay_max_z = np.max(clay_positions[2])

    #inform user of inaccessible clay+interlayer width
    print(f'clay min z is {round(clay_min_z)}, clay max z is {round(clay_max_z)}')
    clay_thickness = clay_max_z - clay_min_z
    print(f'inaccessible layer (2 clay layers and interlayer) is {round(clay_thickness)} Angstroms')

# %%
##### SUBSTITUTION SITES
def isomorphous_substitutions():    
    # make positions global so they can be accessed in analysis() function
    global all_at_x, all_at_y, all_mgo_x, all_mgo_y

    # NOTE:
    # For Al(tet), selects AT atoms up to 2 Angstroms away from the basal clay surface (so only AT in the exposed layer)
    # For Mg(octa), selects MGO atoms up to 4 Angstroms away from basal clay surface

    if side == 'top':
        # TETRAHEDRAL AL
        surface_at_only = clay_max_z - 2    
        at_sel = u.select_atoms(f'name AT* and (prop z <= {clay_max_z} and 'f'prop z >= {surface_at_only})')
        
        # OCTAHEDRAL MG
        surface_mgo_only = clay_max_z - 4    
        mgo_sel = u.select_atoms(f'name MGO* and (prop z <= {clay_max_z} and 'f'prop z >= {surface_mgo_only})')
    
    elif side == 'bottom':
        # TETRAHEDRAL AL
        surface_at_only = clay_min_z + 2    
        at_sel = u.select_atoms(f'name AT* and (prop z >= {clay_min_z} and 'f'prop z <= {surface_at_only})')
        
        # OCTAHEDRAL MG        
        surface_mgo_only = clay_min_z + 4    
        mgo_sel = u.select_atoms(f'name MGO* and (prop z >= {clay_min_z} and 'f'prop z <= {surface_mgo_only})')

    ########################################
    ##### SAVE AND MANIPULATE AT COORDINATES
    at_sel_pos = at_sel.positions    
    at_pos = np.empty((0,3))
    at_pos = np.vstack((at_pos, at_sel_pos))

    # choose only x and y positions
    at_xy = at_pos[0:,0:2]
    all_at_x = np.transpose(at_xy)[0]
    all_at_y = np.transpose(at_xy)[1]

    ##########################################
    ##### SAVE AND MANIPULATE MGO COORDINATES
    mgo_sel_pos = mgo_sel.positions
    mgo_pos = np.empty((0,3))
    mgo_pos = np.vstack((mgo_pos, mgo_sel_pos))

    # store MGO positions in np array for later
    mgo_xy = mgo_pos[0:,0:2]
    
    # select only x and y coordinates
    all_mgo_x = np.transpose(mgo_xy)[0]
    all_mgo_y = np.transpose(mgo_xy)[1]

    ##########################################
    # SAVE SUBSTITUTION XY COORDINATES TO CSV?
    if csv == 'yes':
    # convert array into dataframe 
        at_xy_df = pd.DataFrame(at_xy)
        mgo_xy_df = pd.DataFrame(mgo_xy)  
    # save the dataframe as a csv file 
        at_xy_df.to_csv("at_xy_top.csv")
        mgo_xy_df.to_csv("mgo_xy_top.csv")
    else:
        print('Positions not being saved')  

##### ANALYSIS
def analysis():
    for i in sel:
        # create updating/dynamic atom selection depending on chosen side
        if side == 'top':
            start_z = clay_max_z + z0
            end_z = start_z + dz
            dynamic_sel = u.select_atoms(f'{i} and (prop z >= {start_z} and 'f'prop z <= {end_z})', updating = True)
        if side == 'bottom':
            start_z = clay_min_z - z0
            end_z = start_z - dz
            dynamic_sel = u.select_atoms(f'{i} and (prop z <= {start_z} and 'f'prop z >= {end_z})', updating = True)

        # create empty array to fill with coordinates
        pos = np.empty((0,3))

        # ANALYSIS PERFORMED HERE    
        # iterate through trajectory between selected frames
        for ts in u.trajectory[frame_start:frame_stop]:
            dynamic_sel # run the dynamic selection defined earlier
            pos_dyn = dynamic_sel.positions # record positions of atoms in selection at that frame
                        
            # for each atom (row) in pos_dyn, vertically stack atom positions
            for j in pos_dyn: 
                pos = np.vstack((pos, pos_dyn))
            
            # rinse and repeat
            u.trajectory.next

        ### FILTER THROUGH RECORDED COORDINATES ###
        # select just the x and y coordinates
        pos_xy = pos[0:,0:2]

        # divide x and y into separate arrays
        pos_all_x = np.transpose(pos_xy)[0]
        pos_all_y = np.transpose(pos_xy)[1]

        # SAVING TO CSV
        if csv == 'yes':
            print(f'Saving {i} xy coordinates into csv file')
            # convert array into dataframe 
            pos_xy_df = pd.DataFrame(pos_xy) 
            
            # save the dataframe as a csv file 
            pos_xy_df.to_csv(f"{i}_xy_top.csv")
        else:
            print('Positions not being saved')        

        #### PLOTTING ####

        # if plot_type == 'heatmap':
        fig, ax = plt.subplots(figsize = (5,5), dpi=200)
#            norm = mpl.colors.Normalize(vmin=0, vmax=10)
        if plot_type == 'contourf':
            xx, yy = np.mgrid[minX:maxX:150j, minY:maxY:150j]
            import scipy.stats as st
            positions = np.vstack([xx.ravel(), yy.ravel()])
            values = np.vstack([pos_all_x, pos_all_y])
            kernel = st.gaussian_kde(values)
            kernel.set_bandwidth(bw_method=0.05)
            f = np.reshape(kernel(positions).T, xx.shape)
            plt.contourf(xx, yy, f, cmap='viridis', zorder=1, alpha=1, levels = 20, vmin = 0, vmax = 0.60)
        elif plot_type =='contour':
            xx, yy = np.mgrid[minX:maxX:150j, minY:maxY:150j]
            import scipy.stats as st
            positions = np.vstack([xx.ravel(), yy.ravel()])
            values = np.vstack([pos_all_x, pos_all_y])
            kernel = st.gaussian_kde(values)
            kernel.set_bandwidth(bw_method=0.05)
            f = np.reshape(kernel(positions).T, xx.shape)
            plt.contour(xx, yy, f, cmap='viridis', zorder=1, alpha=1, levels = 20, vmin = 0, vmax = 0.60)
        elif plot_type == 'scatter':
            plt.scatter(pos_all_x, pos_all_y, alpha=0.1, label=f'{i}', marker="x", linewidths=1, color = 'red')
        elif plot_type == 'heatmap':
            norm = mpl.colors.Normalize(vmin=0, vmax=10)
            plt.hist2d(pos_all_x, pos_all_y, bins=300, norm=norm, cmap='viridis', range=[[minX, maxX],[minY, maxY]])
        plt.colorbar()
        plt.scatter(all_mgo_x, all_mgo_y, alpha=1, label='MGO', marker="^", color='cyan', linewidths=4, zorder=4)
        plt.scatter(all_at_x, all_at_y, alpha=1, label='AT', marker="o", color='salmon', linewidths=4, zorder=10)               

        # axis legend
        ax.legend(loc='upper left')
        
        # axis labels
        ax.set_xlabel('x (Å)', fontsize=10)
        ax.set_ylabel('y (Å)', fontsize=10)
        
        # axis titles
        ax.set_title(f'{i} {z0}-{dz} Å')
        
        # set axis limits
        ax.set_xlim(minX, maxX)
        ax.set_ylim(minY, maxY)

        # SAVING     
        # define plot title
        plot_title = f'{i}_SDM_z{z0}to{dz}_frame{frame_start}to{frame_stop}_{plot_type}_top'
        #replace whitespaces with underscores
        plot_title = plot_title.replace(' ','_')
        #save figure
        plt.savefig(f'{plot_title}.png', bbox_inches='tight')

# run the code
setup()
isomorphous_substitutions()
analysis()


#end_time = time.time()

# record execution time for benchmarking
# execution_time = end_time - start_time
# with open('surfdensmap.log', 'a') as write_time:
#     write_time.write(f'{execution_time}')