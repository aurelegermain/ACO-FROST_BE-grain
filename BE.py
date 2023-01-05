# %%


#----------------------------------------------------------------------------------
#Compute binding energies of a molecule around a grain .xyz structure using GFN-xTB
#
#Aurèle Germain
#email aureleroger.germain@unito.it
#----------------------------------------------------------------------------------

from numpy.lib.function_base import blackman
import pandas as pd
from copy import deepcopy
import subprocess
import numpy as np
from scipy.spatial import ConvexHull
from ase import io, Atoms, neighborlist, geometry
from ase.units import kJ,Hartree,mol
from ase.build import molecule
import argparse
import math
import os
from itertools import combinations

rng = np.random.default_rng()

parser = argparse.ArgumentParser()
parser.add_argument("grain", metavar="G", help="Grain model to sample in .xyz", type=str)
parser.add_argument("-l", "--level", help="Grid level needed", type=int, default="0")
parser.add_argument("-sl", "--section_level", help="Section Grid level needed", type=int, default="0")
parser.add_argument("-mol", "--molecule", help="molecule to sample", type=str, default="0", required=True)
parser.add_argument("-g", "--gfn", help="GFN-xTB method to use (0,1,2, or ff)", default="2")
parser.add_argument("-r", "--radius", help="Radius for unfixing molecules", type=float, default="5")
parser.add_argument("-om","--othermethod", action='store_true', help="Other method without the fixing of anything. Temporary name.")
parser.add_argument("-restart", "--restart", help="If the calculation is a restart and from where it restart", type=int, default="0")
parser.add_argument("-range", "--range", nargs='+', help="To contains the computation to only a range of BE", type=int)
parser.add_argument("-rotation", "--rotation", help="If you want more than one random rotation for each adsorption", type=int, default=1)
parser.add_argument("-gc", "--grid_continue", nargs='+', help="If you want to increase the size of the grid. First number is the first level of grid used, second number is the level desired", type=int, default=[0,0])
parser.add_argument("-add_rotation", "--add_rotation", nargs='+', help="If you want to add a different rotation for your adsorption. First number how many rotation you have (default equal 1) and second number is how many rotation you want (2 if you want to add 1 rotation, 3 if you want to add 2, etc..)", type=int, default=[1,1])
parser.add_argument("-d", "--distance", help="distance from grain when projected", type=float, default="2.5")

#Conflicting options part

#conflict between -onlyfreq and -nofreq
group_freq = parser.add_mutually_exclusive_group()
group_freq.add_argument('-nfreq', '--no_freq', action='store_true', help="No frequencies computation")
only_freq_arg = group_freq.add_argument('-ofreq', '--only_freq', action='store_true', help="Only the frequencies computation. Needs the fixed or unfixed files")

#conflict between -onlyfixed and -nofixed
group_fixed = parser.add_mutually_exclusive_group()
group_fixed.add_argument('-nf', '--no_fixed', action='store_true', help="The grain is totally unfixed")
only_fixed_arg = group_fixed.add_argument('-of', '--only_fixed', action='store_true', help="Only the opt of the fixed grain")

#conflict between -onlyfixed -onlyunfixed and -onlyfreq
group_only = parser.add_mutually_exclusive_group()
group_only.add_argument('-ouf', '--only_unfixed', action='store_true', help="Only the opt of the unfixed grain. Needs the fixed files")
group_only._group_actions.append(only_freq_arg)
group_only._group_actions.append(only_fixed_arg)

parser.add_argument('-nu', '--no_unfixed', action='store_true', help="The unfixed part is not computed.")


args = parser.parse_args()

level = args.level
section_level = args.section_level
grain = args.grain
molecule_to_sample = args.molecule.upper() #The molecule indicated by the user is automatically put in upper case 
distance_grain = args.distance
gfn = str(args.gfn)
radius_unfixed = args.radius
nofreq =args.no_freq
nofixed = args.no_fixed
nounfixed = args.no_unfixed
onlyfixed = args.only_fixed
onlyunfixed = args.only_unfixed
onlyfreq = args.only_freq
othermethod = args.othermethod
restart = args.restart
list_args_grid_continue = args.grid_continue
list_range = args.range
nbr_rotation = args.rotation
add_rotation = args.add_rotation

#parameters needed for the starting positions. 
#Should be put inside the function at some point (not used anywhere else I think)
distance = 2.5
coeff_min = 1.00
coeff_max = 1.2
steps = 0.1

if add_rotation != [1,1]:
    list_args_grid_continue = [level,level]
    nbr_rotation = add_rotation[1]

#This is the file that contains the id of the BE that had opt problems in fixed, or are too far from the grain in unfixed
list_discarded = "list_discarded.txt"

def grid_continue(starting_grid, ending_grid):
    """ 
    If the user wants to increase the number of BEs by increasing the size of the grid. 
    Takes the level of the old grid and the level of the desired one and return the number of BE to ignore so that as to not compute the old grid again
    """
    print(starting_grid, ending_grid)
    if (ending_grid - starting_grid) == 0 and (add_rotation[1] - add_rotation[0]) == 0:
        continue_grid = 0
    else:
        list_size_grid = np.atleast_1d(np.zeros(ending_grid + 1))
        for i in range(len(list_size_grid)):
            if i == 0:
                list_size_grid[i] = 12
            else:
                list_size_grid[i] = list_size_grid[i - 1] + 2**(2*(i + 1) + 1) - 2**(2*(i - 1) + 1) #this is the equation that gives the number of grid points for the level n (depends on N(n-1))
        continue_grid = int(list_size_grid[starting_grid]*add_rotation[0])
    
    print(continue_grid)


    return continue_grid

def grid_building(sphere, level, continue_grid):
    """
    Build the desired level of grid from the level 0.
    When the desired level is reached, each grid point is replaced by the molecule studied (with random orientation) and brought closer to the grain structure for the future opt.
    Ask for the grain (sphere) structure in Atoms object from the ase module, the grid level desired by the user, and a value named "continue grid" which is the values computed by the grid_continue function (default is 0 if the grid is not increased from a previous calculation)
    
    Grid taken from N. A. Teanby. 2006. Comput. Geosci. 32, 9 (November, 2006), 1442–1450. DOI:https://doi.org/10.1016/j.cageo.2006.01.007
    """


    a = 2*np.cos(np.pi/5)
    unit_sphere = np.sqrt(1 + 4*np.square(np.cos(np.pi/5)))
    seed = np.array((0,a,1))
    grid = np.zeros((12,3))

    #level 0 of the grid that we need to define
    grid[0,:] = [0,a,1]
    grid[1,:] = [0,-a,1]
    grid[2,:] = [0,a,-1]
    grid[3,:] = [0,-a,-1]
    grid[4,:] = [1,0,a]
    grid[5,:] = [-1,0,a]
    grid[6,:] = [1,0,-a]
    grid[7,:] = [-1,0,-a]
    grid[8,:] = [a,1,0]
    grid[9,:] = [-a,1,0]
    grid[10,:] = [a,-1,0]
    grid[11,:] = [-a,-1,0]

    #projection onto the unit sphere
    grid = grid/unit_sphere

    iter_level = 0
    #we go level by level. If level 2 is chosen we build the level 1 from level 0 first
    while iter_level<level:
        #build the indice of each triangles
        hull = ConvexHull(grid)
        indices = hull.simplices
    
        #sort indices inside each row and then we sort each row dependng on the value of the first column
        indices = np.sort(indices)
        indices =indices[indices[:, 0].argsort()]
    
        #build the list of each point to be added to construct the next level grid
        list_point_to_add= []
        for l in range(len(indices)):
            list_point_to_add = np.append(list_point_to_add,[i for i in combinations(indices[l,:], 2)])
        list_point_to_add = np.reshape(list_point_to_add, (-1,2))
        list_point_to_add = np.unique(list_point_to_add, axis=0).astype(int)
        
        #build the grid
        grid_to_append = np.zeros([len(list_point_to_add),3])
        for i in range(len(grid_to_append)):
            grid_to_append[i,:] = grid[list_point_to_add[i,0],:] + grid[list_point_to_add[i,1],:]
            grid_to_append[i,:] = grid_to_append[i,:]/np.sqrt(np.sum(np.square(grid_to_append[i,:])))

        #we add the new point to the previous grid
        grid = np.append(grid, grid_to_append, axis=0)

        iter_level +=1
    
     #if we bisect or trisect or ... the grid level we chosed 
    if section_level > 0:
        #the first part is the same as for building the grid levels
        #build the indice of each triangles
        hull = ConvexHull(grid)
        indices = hull.simplices
        #sort indices inside each row and then we sort each row dependng on the value of the first column
        indices = np.sort(indices)
        indices =indices[indices[:, 0].argsort()]
    
        #build the list of each point to be added to construct the next level grid
        list_point_to_add= []
        for l in range(len(indices)):
            list_point_to_add = np.append(list_point_to_add,[i for i in combinations(indices[l,:], 2)])
        list_point_to_add = np.reshape(list_point_to_add, (-1,2))
        list_point_to_add = np.unique(list_point_to_add, axis=0).astype(int)

        #we make a matrice for each point. 0 if two points are not neighbours and 1 if they are
        matrice_connect = np.zeros([len(grid), len(grid)])
        for connect in list_point_to_add:
            matrice_connect[connect[0], connect[1]] = 1
            matrice_connect[connect[1], connect[0]] = 1

        #we go points by points 
        for i, line in enumerate(matrice_connect):

            #list of points connected to the one we selected 
            list_points = [i for i in range(len(line)) if line[i]==1]

            #this array will contain every section points between the selected point and the ones in the list
            grid_direct = np.zeros([section_level*len(list_points),3])
            for j, point in enumerate(list_points):
                if point < i: continue

                #we create the sections points for every point directly connected to the selected point 
                for lvl in range(section_level):
                    m = lvl + 1 #m equal 1 if bisection, 2 then 1 if trisection, 3 then 2 then 1 if quadrisection, etc...
                    n = (section_level + 1) - m 
                    
                    grid_direct[j*section_level + lvl,:] = (n * grid[i,:] + m * grid[point,:])/(n + m)
                    grid_direct[j*section_level + lvl,:] = grid_direct[j*section_level + lvl,:]/np.sqrt(np.sum(np.square(grid_direct[j*section_level + lvl,:]))) 
            #we append these points to the grid
            grid = np.append(grid, grid_direct[np.any(grid_direct != [0.,0.,0.], axis=1)], axis=0)
            
            #in this part we will create the 
            for j, point in enumerate(list_points):
                if point < i: continue

                list_connect_point = [k for k in range(len(line)) if (line[k]==1 and matrice_connect[point][k] == 1)]

                for k, connect_point in enumerate(list_connect_point):
                    if connect_point < point: continue
                    grid_connect = np.zeros([sum([i + 1 for i in range(section_level - 1)]),3])

                    iter = 0
                    for lvl in range(section_level-1):
                        for nbr_lvl in range(lvl + 1):
                            m = nbr_lvl + 1
                            n = (lvl + 2) - m 

                            grid_connect[iter,:] = (n * grid_direct[j*section_level + lvl + 1,:] + m * grid_direct[section_level*list_points.index(connect_point) + lvl + 1,:])/(n + m)
                            grid_connect[iter,:] = grid_connect[iter,:]/np.sqrt(np.sum(np.square(grid_connect[iter,:]))) 
                            iter +=1

                    grid = np.append(grid, grid_connect, axis=0)

    matrice_distances = geometry.get_distances(sphere.get_positions(), sphere.get_center_of_mass())
    list_distances = [values for list in matrice_distances[1] for values in list if values != 0]

    grid_list = Atoms(str(len(grid)) + 'N')
    grid_list.set_positions(grid*(np.amax(list_distances) + 2.5))

    del grid
    
    if nbr_rotation !=1:
        grid_deepcopy = deepcopy(grid_list)
        for i in range(nbr_rotation - 1):
            grid_list = np.append(grid_list, grid_deepcopy, axis=0)

    #this is simply to project on a sphere with a radius taken from the grain model to sample
    distance_max = np.amax(distances_3d(sphere)) + 1

    #Build an ase Atoms object using the studied molecule. Then perform an xtb opt and a frequency calculation. The opt geometry will be used for the input file, and the frequency calculation will be used for ZPE corrected BE 
    if continue_grid == 0: #If we continue from a previsous grid this part was already done
        #make a directory with the name of the molecule + _xtb. To store the xtb files of the molecule to sample
        subprocess.call(['mkdir', molecule_to_sample + '_xtb'])

        io.write('./' + molecule_to_sample + '_xtb/' + molecule_to_sample + '_inp.xyz', molecule_csv(molecule_to_sample))
        start_GFN(gfn,'extreme', molecule_to_sample + '_inp.xyz', "output",molecule_to_sample + '_xtb',0)
        #process = subprocess.Popen(['xtb', molecule_to_sample + '_inp.xyz', '--opt', 'extreme', '--gfn' + gfn, '--verbose'], cwd='./' + molecule_to_sample + '_xtb', stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        #stdout, stderr = process.communicate()
        #output = open("./" + molecule_to_sample + "_xtb/output", "w")
        #print(stdout.decode(errors="replace"), file=output)
        #print(stderr.decode(errors="replace"), file=output)   
        #output.close()

        
        subprocess.call(['mv', './' + molecule_to_sample + '_xtb/xtbopt.xyz', './' + molecule_to_sample + '_xtb/' + molecule_to_sample + '.xyz'])
        
        start_GFN_freq(gfn, molecule_to_sample + '.xyz', 'frequencies', molecule_to_sample + '_xtb', 0)

        #process = subprocess.Popen(['xtb', molecule_to_sample + '.xyz', '--hess', '--gfn' + gfn, '--verbose'], cwd='./' + molecule_to_sample + '_xtb', stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        #stdout, stderr = process.communicate()
        #output = open("./" + molecule_to_sample + "_xtb/frequencies", "w")
        #print(stdout.decode(errors="replace"), file=output)
        #print(stderr.decode(errors="replace"), file=output)   
        #output.close()

    #this is the molecule that will be added for each rid point.
    atoms_to_add = io.read('./' + molecule_to_sample + '_xtb/' + molecule_to_sample + '.xyz') 

    for i in range(len(grid_list) - continue_grid):
        #if i!=0: continue
        i_continue = int(i + continue_grid)
        Atoms_to_add = deepcopy(atoms_to_add)

        #rotate the molecule randomly
        angle_mol = rng.random(3)*360
        Atoms_to_add.rotate(angle_mol[0], "x")
        Atoms_to_add.rotate(angle_mol[1], "y")
        Atoms_to_add.rotate(angle_mol[2], "z")


        Atoms_to_add.set_positions(Atoms_to_add.get_positions() + grid_list[i_continue].position)

        if 'atoms_far' in locals():
            atoms_far += deepcopy(Atoms_to_add)
        else:
            atoms_far = deepcopy(Atoms_to_add)

        good_place = False
        start_again = 0
        while good_place!=True:
            All_distances_from_grain = geometry.get_distances(Atoms_to_add.get_positions(), sphere.get_positions())[1]
            id_mol, id_nearest = np.divmod(np.argmin(All_distances_from_grain),len(sphere))

            mol_position = Atoms_to_add[id_mol].position
            nearest_position = sphere[id_nearest].position

            delta_a = np.sum(mol_position**2)
            delta_b = -2*np.sum(mol_position*nearest_position)
            delta_c = np.sum(nearest_position**2) - distance_grain**2
            delta = delta_b**2 -4*delta_a*delta_c

            if delta > 0:
                t_0 = (-delta_b + np.sqrt(delta))/(2*delta_a)
                t_1 = (-delta_b - np.sqrt(delta))/(2*delta_a)

                mol_position_0 = np.array([t_0*mol_position[0], t_0*mol_position[1], t_0*mol_position[2]])
                mol_position_1 = np.array([t_1*mol_position[0], t_1*mol_position[1], t_1*mol_position[2]])

                d_mol_0 = np.sqrt(np.sum((mol_position_0)**2))
                d_mol_1 = np.sqrt(np.sum((mol_position_1)**2))

                if d_mol_0 > d_mol_1: 
                    diff_mol_position = mol_position_0 - mol_position
                else:
                    diff_mol_position = mol_position_1 - mol_position
            else:
                delta_c_2 = [np.sum(nearest_position**2) - (i/10)**2 for i in range(20,40,1)]
                delta_2 = [delta_b**2 -4*delta_a*delta_c_i for delta_c_i in delta_c_2]

                if start_again > 5:
                    t_2_0 = (-delta_b + np.sqrt(np.abs(delta)))/(2*delta_a)
                    t_2_1 = (-delta_b - np.sqrt(np.abs(delta)))/(2*delta_a)

                    mol_position_2_0 = np.array([t_2_0*mol_position[0], t_2_0*mol_position[1], t_2_0*mol_position[2]])
                    mol_position_2_1 = np.array([t_2_1*mol_position[0], t_2_1*mol_position[1], t_2_1*mol_position[2]])

                    d_mol_2_0 = np.sqrt(np.sum((mol_position_2_0)**2))
                    d_mol_2_1 = np.sqrt(np.sum((mol_position_2_1)**2))

                    if d_mol_2_0 < d_mol_2_1: 
                        diff_mol_position = mol_position_2_0 - mol_position
                    else:
                        diff_mol_position = mol_position_2_1 - mol_position
                else:
                    t_2 = (-delta_b)/(2*delta_a)
                    mol_position_2 = np.array([t_2*mol_position[0], t_2*mol_position[1], t_2*mol_position[2]])
                    diff_mol_position = mol_position_2 - mol_position

            Atoms_to_add_0 = deepcopy(Atoms_to_add)

            Atoms_to_add.set_positions(Atoms_to_add.get_positions() + diff_mol_position)

            list_min_distances_from_grain = [np.argmin(distances) for distances in All_distances_from_grain] + [np.argmin(All_distances_from_grain)]
            All_distances_from_grain_1 = geometry.get_distances(Atoms_to_add.get_positions(), sphere.get_positions())[1]
            if ((np.amin(All_distances_from_grain_1) > distance_grain*0.95) and (np.amin(All_distances_from_grain_1) < distance_grain*1.05)) or start_again > 10:
                good_place = True
                #print(i, start_again, np.amin(All_distances_from_grain_1), delta)
                if 'atoms' in locals():
                    atoms += deepcopy(Atoms_to_add)
                else:
                    atoms = deepcopy(Atoms_to_add)
            else:
                #print(i, start_again, np.amin(All_distances_from_grain_1), delta)
                start_again += 1

    if continue_grid == 0: 
        #writ the file with all the final input positions
        io.write('./grid.xyz', sphere + atoms)
        io.write('./grid_first.xyz', sphere + atoms_far)
    else:
        #if the computation continues from a previous one we update the file
        grid_old = io.read('./grid.xyz') 
        io.write('./grid.xyz', grid_old + atoms)
        grid_first_old = io.read('./grid_first.xyz') 
        io.write('./grid_first.xyz', grid_first_old + atoms_far)

def distances_3d(atoms):
    """Takes an ase atoms object and return their distances from the origin"""
    x = atoms.get_positions()[:,0] if hasattr(atoms, '__len__') else atoms.position[0] #if there is only one atome it used position[] instead of get_positions
    y = atoms.get_positions()[:,1] if hasattr(atoms, '__len__') else atoms.position[1]
    z = atoms.get_positions()[:,2] if hasattr(atoms, '__len__') else atoms.position[2]
    list_distances = np.sqrt(x**2 + y**2 + z**2)
    return list_distances

def distances_ab(atoms_a, atoms_b):
    """Takes two ase atoms objects and return the distance between them."""
    x = atoms_a.get_positions()[:,0] if hasattr(atoms_a, '__len__') else atoms_a.position[0]
    y = atoms_a.get_positions()[:,1] if hasattr(atoms_a, '__len__') else atoms_a.position[1]
    z = atoms_a.get_positions()[:,2] if hasattr(atoms_a, '__len__') else atoms_a.position[2]
    
    u = atoms_b.get_positions()[:,0] if hasattr(atoms_b, '__len__') else atoms_b.position[0]
    v = atoms_b.get_positions()[:,1] if hasattr(atoms_b, '__len__') else atoms_b.position[1]
    w = atoms_b.get_positions()[:,2] if hasattr(atoms_b, '__len__') else atoms_b.position[2]
    
    if hasattr(atoms_a, '__len__') and hasattr(atoms_b, '__len__'):
        if np.shape(atoms_a.get_positions()) != np.shape(atoms_b.get_positions()):
            list_distances = np.zeros((len(x),len(u)))
            for i in range(len(x)):
                list_distances[i,:] = np.sqrt(np.square(x[i]-u) + np.square(y[i]-v) + np.square(z[i]-w))
        else:
            list_distances = np.sqrt(np.square(x-u) + np.square(y-v) + np.square(z-w))
    else:
        list_distances = np.sqrt(np.square(x-u) + np.square(y-v) + np.square(z-w))
    return list_distances

def barycentre(atoms):
    """Takes an ase atoms object and centre it on its barycentre."""
    positions = atoms.get_positions()
    barycentre_position = np.array([np.sum(positions[:,0])/len(positions[:,0]),np.sum(positions[:,1])/len(positions[:,0]),np.sum(positions[:,2])/len(positions[:,0])])
    new_positions = np.zeros(np.shape(positions))
    for i in range(len(new_positions)):
        new_positions[i,:] = positions[i,:] - barycentre_position
    atoms.set_positions(new_positions)
    return atoms

def FromXYZtoDataframeMolecule(input_file):
    """read and encode .xyz file into Pandas Dataframe"""
    df_xyz = pd.DataFrame(columns = ['Atom','X','Y','Z'])
    with open(input_file, "r") as data:
        lines = data.readlines()
        for i in range(2,len(lines)):
            line = lines[i].split()
            if len(line) == 4:
                new_row = pd.Series({'Atom':line[0],'X':line[1],'Y':line[2],'Z':line[3]},name=3)
                df_xyz = df_xyz.append(new_row,ignore_index=True)
    df_xyz['Atom'].astype(str)
    df_xyz['X'].astype(float)
    df_xyz['Y'].astype(float)
    df_xyz['Z'].astype(float)
    #compute neighbor of the atoms in xyz format with ASE package
    mol = io.read(input_file)
    cutOff = neighborlist.natural_cutoffs(mol, mult=0.9)
    neighborList = neighborlist.NeighborList(cutOff, self_interaction=False, bothways=True)
    neighborList.update(mol)
    #create molecules column in dataframe related to connectivity, molecules start from 0
    mol_vector = np.zeros(df_xyz.shape[0],dtype=int)
    for i in range(df_xyz.shape[0]):
        if i == 0:
            mol_ix = 1
            n_list = neighborList.get_neighbors(i)[0]
            mol_vector[i] = mol_ix
            for j,item_j in enumerate(n_list):
                mol_vector[item_j] = mol_ix
                j_list = neighborList.get_neighbors(item_j)[0]
                j_list = list(set(j_list) - set(n_list))
                for k,item_k in enumerate(j_list):
                    mol_vector[item_k] = mol_ix
        elif mol_vector[i] == 0:
            mol_ix = mol_ix + 1 
            n_list = neighborList.get_neighbors(i)[0]
            mol_vector[i] = mol_ix
            for j,item_j in enumerate(n_list):
                mol_vector[item_j] = mol_ix
                j_list = neighborList.get_neighbors(item_j)[0]
                j_list = list(set(j_list) - set(n_list))
                for k,item_k in enumerate(j_list):
                    mol_vector[item_k] = mol_ix
    mol_vector = mol_vector - 1
    df_xyz['Molecules'] = mol_vector
    return(df_xyz)

def LabelMoleculesRadius(df_xyz,mol_ref,radius):
    """Indicates which molecules are inside the unfixed radius."""
    df_xyz['X'].astype(float)
    df_xyz['Y'].astype(float)
    df_xyz['Z'].astype(float)
    n_mol = df_xyz['Molecules'].max() + 1
    mol_vector = np.array(list(range(n_mol)),dtype=int)
    X_c = np.zeros(n_mol,dtype=float)
    Y_c = np.zeros(n_mol,dtype=float)
    Z_c = np.zeros(n_mol,dtype=float)
    for i in range(n_mol):
        df_tmp = df_xyz[df_xyz['Molecules'] == i]
        X_c[i] = df_tmp['X'].astype(float).mean()
        Y_c[i] = df_tmp['Y'].astype(float).mean()
        Z_c[i] = df_tmp['Z'].astype(float).mean()
    df_center = pd.DataFrame()
    df_center['Molecules'] = mol_vector
    df_center['X'] = X_c
    df_center['Y'] = Y_c
    df_center['Z'] = Z_c
    tmp_c = df_center[df_center['Molecules'] == mol_ref].values[0]
    dist_vector = np.zeros(n_mol,dtype=float)
    dist_bool = np.full(n_mol, False)
    for index, rows in df_center.iterrows():
        dist_vector[index] = math.sqrt((rows.X - tmp_c[1])**2 + (rows.Y - tmp_c[2])**2 + (rows.Z - tmp_c[3])**2)
        if dist_vector[index] < radius:
            dist_bool[index] = True
    df_center['Distance'] = dist_vector
    df_center['Shell'] = dist_bool
    xyz_bool = np.full(df_xyz.shape[0], 'M')
    for index, rows in df_xyz.iterrows():
        if df_center[df_center['Molecules'] == rows.Molecules].values[0][5] == True:
            xyz_bool[index] = 'H'    
    df_xyz['Level'] = xyz_bool
    return(df_xyz)

#Start of the grain totally fixed part of BE computation
def fixed(restart, continue_grid, list_range):
    """
    Read the grid.xyz file and produce separate input files for each position. 
    Then compute the GFN-xTB opt of the inputs and use an xtb.inp file to fix the grain.
    """
    #Create every input for the fixed part
    for i in range(len_grid):
        if i < continue_grid:
            continue
        if i < list_range[0] or i > list_range[1]:
            continue 
        subprocess.call(['mkdir', str(i)])
        file_xtb_input = open("./" + str(i) + "/xtb.inp","w")
        print("$constrain", file=file_xtb_input)
        print("    atoms: 1-" + str(len_sphere), file=file_xtb_input)
        print("$end", file=file_xtb_input)
        file_xtb_input.close()
        io.write('./' + str(i) + '/BE_' + str(i) + '.xyz', sphere + sphere_grid[len_sphere + i*len_molecule:len_sphere + (i+1)*len_molecule])
    
    #Compute the BE with the grain totally fixed
    for i in range(len_grid):
        if i < restart or i < continue_grid:
            continue
        if i < list_range[0] or i > list_range[1]:
            continue 
        
        start_GFN(gfn, 'extreme', 'BE_' + str(i) + '.xyz', "BE_" + str(i) + ".out", str(i), 'xtb.inp')
        
        #process = subprocess.Popen(['xtb', '--input', 'xtb.inp', 'BE_' + str(i) + '.xyz', '--opt', 'extreme', '--gfn' + gfn, '--verbose'], cwd='./' + str(i), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        #stdout, stderr = process.communicate()
        #output = open("./" + str(i) + "/BE_" + str(i) + ".out", "w")
        #print(stdout.decode(errors="replace"), file=output)
        #print(stderr.decode(errors="replace"), file=output)   
        #output.close()
        subprocess.call(['mv', './' + str(i) + '/xtbopt.log', './' + str(i) + '/movie.xyz'])

        #if the computation has not converged the BE is added to a file indicating every BE that did not work and should not be used for further computations.
        if os.path.isfile("./" + str(i) + "/NOT_CONVERGED"):
            file_list_discarded = open(list_discarded, "a")
            print(str(i) + "    N", file=file_list_discarded)
            file_list_discarded.close()

def unfixed(restart, continue_grid, list_range):
    """
    Compute the GFN-xTB opt of the xtbopt.xyz file obtained during the "fixed" function.
    Unfix molecules around a indicated radius around the molecule studied (5A by default).
    If after opt the molecules present in the radius have changed, it unfixes the new list and perform a new opt.
    This process continue until the list stays the same (converges).
    An output file is produced containing the id of each BE and a letter Y or N. 
    "Y" means the computation worked, "N" means the BE is discarded for further computation (frequencies).
    """
    #Chech if the list_discarded file exist. If yes it opens it and create a list of good BE. If no then the list of good BE contains all the BE
    try:
        open(list_discarded, "r")
    except FileNotFoundError:
        list_not_discarded = np.arange(len_grid)
    else:
        list_not_discarded = np.arange(len_grid)
        file_list_discarded = np.loadtxt(list_discarded, dtype=str)
        file_list_discarded = np.atleast_2d(file_list_discarded)
        list_discarded_array = file_list_discarded[:,0].astype(int)
        list_not_discarded = np.setdiff1d(list_not_discarded, list_discarded_array)

    #produces the input part from the xtbopt.xyz file of the "fixed" function. 
    # If we are doing a restart then this part is not done.  
    for i in range(len_grid):
        if restart > 0: 
            break
        if i < continue_grid or i not in list_not_discarded:
            continue
        if i < list_range[0] or i > list_range[1]:
            continue 
        if os.path.isdir('./' + str(i) + '/') is False:
            print('Folder with geometry needed not found')
            exit()
        subprocess.call(['mkdir', './' + str(i) + '/unfixed-radius'])
        subprocess.call(['cp', './' + str(i) + '/xtbopt.xyz', './' + str(i) + '/unfixed-radius/BE_' + str(i) + '.xyz'])
    
        input_BE_unfixed  = './' + str(i) + '/unfixed-radius/BE_' + str(i) + '.xyz'
        df_xyz_test = FromXYZtoDataframeMolecule(input_BE_unfixed) 
        mol_ref = df_xyz_test['Molecules'].max()
        df_xyz_test = LabelMoleculesRadius(df_xyz_test,mol_ref,radius_unfixed)
        grain_structure = df_xyz_test.to_numpy()

        #produce the list of fixed atoms depending on the LabelMoleculesRadius function
        for k in range(len(grain_structure[:,5])):
            if grain_structure[k,5] == "M":
                if "list_fix" in locals():
                    list_fix = np.append(list_fix,k)
                else:
                    list_fix = k
            elif grain_structure[k,5] == "H" and grain_structure[k,4]!=mol_ref:
                if "list_not_fix" in locals():
                    list_not_fix = np.append(list_not_fix,k)
                else:
                    list_not_fix = k

        #Produces the xtb.inp file indicating which atom to fix
        file_xtb_unfixed_input = open("./" + str(i) + "/unfixed-radius/xtb.inp","w")
        print("$constrain", file=file_xtb_unfixed_input)
        print("    atoms: ", end="", file=file_xtb_unfixed_input)
        #print(list_fix)
        list_fix = np.atleast_1d(list_fix)
        for k in range(len(list_fix)):
            if k!=0:
                if k==len(list_fix)-1:
                    if last_fix == k - 1:
                        print("-" + str(list_fix[k]+1), end="", file=file_xtb_unfixed_input)
                        j = j + 1
                    else:
                        print("," + str(list_fix[k]+1), end="", file=file_xtb_unfixed_input)
                else:
                    if list_fix[last_fix] == list_fix[k] - 1:
                        last_fix = k
                    else:
                        print("-" + str(list_fix[last_fix]+1) + "," + str(list_fix[k]+1), end="", file=file_xtb_unfixed_input)
                        last_fix = k
                        j = j + 1
            elif k==0:
                j = 0
                last_fix = k
                print(list_fix[k]+1, end="", file=file_xtb_unfixed_input)
    
        print("\n$end", file=file_xtb_unfixed_input)
        file_xtb_unfixed_input.close()
    
        #Check if the list of not fixed molecules exist and delete it if yes (for the next BE) if no it means that the molecule is too far from the grain and thus is added to the discarded file
        if "list_not_fix" in locals():
            del list_not_fix
        else:
            file_list_discarded = open(list_discarded, "a")
            print(str(i) + "    N", file=file_list_discarded)
            file_list_discarded.close()
    
        if "list_fix" in locals(): 
            del list_fix
    #Read again the discarded file to build the list of relevant BE
    try:
        open(list_discarded, "r")
    except FileNotFoundError:
        list_not_discarded = np.arange(len_grid)
    else:
        list_not_discarded = np.arange(len_grid)
        file_list_discarded = np.loadtxt(list_discarded, dtype=str)
        file_list_discarded = np.atleast_2d(file_list_discarded)
        list_discarded_array = file_list_discarded[:,0].astype(int)
        list_not_discarded = np.setdiff1d(list_not_discarded, list_discarded_array)
    
    #Part that computes the GFN-xTB opt with the unfixed radius.
    for i in range(len_grid):
        if i < restart or i < continue_grid:
            continue
        if i < list_range[0] or i > list_range[1]:
            continue 
        if i in list_not_discarded:
            
            start_GFN(gfn, 'extreme', 'BE_' + str(i) + '.xyz', "BE_" + str(i) + ".out", str(i) + '/unfixed-radius', 'xtb.inp')
            
            #process = subprocess.Popen(['xtb', '--input', 'xtb.inp', 'BE_' + str(i) + '.xyz', '--opt', 'extreme', '--gfn' + gfn, '--verbose'], cwd='./' + str(i) + '/unfixed-radius', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
            #stdout, stderr = process.communicate()
            #output = open("./" + str(i) + "/unfixed-radius/BE_" + str(i) + ".out", "w")
            #print(stdout.decode(errors="replace"), file=output)
            #print(stderr.decode(errors="replace"), file=output)   
            #output.close()
            subprocess.call(['mv', './' + str(i) + '/unfixed-radius/xtbopt.log', './' + str(i) + '/unfixed-radius/movie.xyz'])
    
            restart_BE = 0
            equal_structure = False 
    
            #This part check if the list of unfixed molecule from the xtbopt.xyz structure is the same as the previous input. 
            # If yes equal_structure = True and it stops, if no then equal_structure = False and the xtbopt.xyz becomes the new input.
            while equal_structure!=True:
                #if there is a problem during the opt the BE is added to the discarded BE list
                if os.path.isfile("NOT_CONVERGED"):
                    equal_structure = True
                    Results = open("./results_lorenzo_unfixed_grain.txt", "a")
                    print(str(i) + " N", file=Results)
                    Results.close()
                    
                    file_list_discarded = open(list_discarded, "a")
                    print(str(i) + "    N", file=file_list_discarded)
                    file_list_discarded.close()
                    continue
                
                #opening of the input and the xtbopt and computation of the unfixed list in both to compare them.
                input_file  = './' + str(i) + '/unfixed-radius/BE_' + str(i) + '.xyz'
                df_xyz_test2 = FromXYZtoDataframeMolecule(input_file) 
                mol_ref2 = df_xyz_test2['Molecules'].max()
                df_xyz_test2 = LabelMoleculesRadius(df_xyz_test2,mol_ref2,radius_unfixed)
                grain_structure2 = df_xyz_test2.to_numpy()
        
                output_file  = './' + str(i) + '/unfixed-radius/xtbopt.xyz'
                df_xyz_test = FromXYZtoDataframeMolecule(output_file) 
                mol_ref = df_xyz_test['Molecules'].max()
                df_xyz_test = LabelMoleculesRadius(df_xyz_test,mol_ref,radius_unfixed)
                grain_structure = df_xyz_test.to_numpy()

                #if the list are different xtbopt.xyz becomes the input and xtb.inp contains the new list of unfixed molecule.
                #This process is done until the old list and the new list are the same.
                if np.array_equal(grain_structure[:,5],grain_structure2[:,5])==False:
                    subprocess.call(['mv', './' + str(i) + '/unfixed-radius', './' + str(i) + '/' + str(restart_BE)])
                    subprocess.call(['mkdir', './' + str(i) + '/unfixed-radius'])
                    subprocess.call(['cp', './' + str(i) + '/' + str(restart_BE) + '/xtbopt.xyz', './' + str(i) + '/unfixed-radius/BE_' + str(i) + '.xyz'])
        
                    for k in range(len(grain_structure[:,5])):
                        if grain_structure[k,5] == "M":
                            if "list_fix" in locals():
                               list_fix = np.append(list_fix,k)
                            else:
                                 list_fix = k
                        elif grain_structure[k,5] == "H" and grain_structure[k,4]!=mol_ref:
                            if "list_not_fix" in locals():
                                list_not_fix = np.append(list_not_fix,k)
                            else:
                                list_not_fix = k
        
                    file_xtb_unfixed_input = open("./" + str(i) + "/unfixed-radius/xtb.inp","w")
                    print("$constrain", file=file_xtb_unfixed_input)
                    print("    atoms: ", end="", file=file_xtb_unfixed_input)
                    #print(list_fix)
                    for k in range(len(list_fix)):
                        if k!=0:
                            if k==len(list_fix)-1:
                                if last_fix == k - 1:
                                    print("-" + str(list_fix[k]+1), end="", file=file_xtb_unfixed_input)
                                    m = m + 1
                                else:
                                    print("," + str(list_fix[k]+1), end="", file=file_xtb_unfixed_input)
                            else:
                                if list_fix[last_fix] == list_fix[k] - 1:
                                    last_fix = k
                                else:
                                    print("-" + str(list_fix[last_fix]+1) + "," + str(list_fix[k]+1), end="", file=file_xtb_unfixed_input)
                                    last_fix = k
                                    m = m + 1
                        elif k==0:
                            m = 0
                            last_fix = k
                            print(list_fix[k]+1, end="", file=file_xtb_unfixed_input)
                
                    print("\n$end", file=file_xtb_unfixed_input)
                    file_xtb_unfixed_input.close()
    
                    del list_fix
                    del list_not_fix
    
                    start_GFN(gfn, 'extreme', 'BE_' + str(i) + '.xyz', "BE_" + str(i) + ".out", str(i) + '/unfixed-radius', 'xtb.inp')
                
                    #process = subprocess.Popen(['xtb', '--input', 'xtb.inp', 'BE_' + str(i) + '.xyz', '--opt', 'extreme', '--gfn' + gfn, '--verbose'], cwd='./' + str(i) + '/unfixed-radius', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
                    #stdout, stderr = process.communicate()
                    #output = open("./" + str(i) + "/unfixed-radius/BE_" + str(i) + ".out", "w")
                    #print(stdout.decode(errors="replace"), file=output)
                    #print(stderr.decode(errors="replace"), file=output)   
                    #output.close()
                    subprocess.call(['mv', './' + str(i) + '/unfixed-radius/xtbopt.log', './' + str(i) + '/unfixed-radius/movie.xyz'])  
    
                    restart_BE = restart_BE + 1
                else:
                    equal_structure = True
            if restart_BE > 0:
                for l in range(restart_BE):
                    subprocess.call(['mv', './' + str(i) + '/' + str(l), './' + str(i) + '/unfixed-radius']) #every time the list is changed the files are put in a numbered folder and put inside the unfixed-radius folder.
            if os.path.isfile("NOT_CONVERGED") == False:
                Results = open("./results_lorenzo_unfixed_grain.txt", "a")
                print(str(i) + " Y", file=Results)
                Results.close()
            else:
                file_list_discarded = open(list_discarded, "a")
                print(str(i) + "    N", file=file_list_discarded)
                file_list_discarded.close()

def frequencies(restart, othermethod, continue_grid, list_range):
    """
    Compute the GFN-xTB frequencies of each BE.
    If the frequencies are computed from an unfixed computation it add the xtb.inp file containing the fixed molecules of the grain.
    For unfixed computation it also computes the frequencies of the grain structure and uses the xtb.inp of BE calculation on the grain structure.
    For the "other method" it computes the freauencies of each BEs normally and compute also the frequencies of the grain (only one time).
    An output file is produced with "N" if the BE of the grain present imaginary frequencies, and "B" is the computation worked well.
    """

    try:
        open(list_discarded, "r")
    except FileNotFoundError:
        list_not_discarded = np.arange(len_grid)
    else:
        list_not_discarded = np.arange(len_grid)
        file_list_discarded = np.loadtxt(list_discarded, dtype=str)
        file_list_discarded = np.atleast_2d(file_list_discarded)
        list_discarded_array = file_list_discarded[:,0].astype(int)
        list_not_discarded = np.setdiff1d(list_not_discarded, list_discarded_array)

    #Do first the othermethod computation if it is asked or find the appropriate files.
    grain_freq = False
    #Compute the frequencies of the grain structure one time. 
    #I don't think the "for" is needed. I should probably take time to better think through this part.
    for i in range(len_grid):
        if i < list_range[0] or i > list_range[1]:
            continue 
        if i in list_not_discarded:
            if os.path.isdir('./' + str(i) + '/') is True and os.path.isfile('./' + str(i) + '/xtb.inp') is False and grain_freq is False:
                othermethod = True
                if continue_grid == 0 or restart == 0:
                    start_GFN_freq(gfn, 'xtbopt.xyz', 'grain_frequencies.out', 'grain', 0)
                    #process = subprocess.Popen(['xtb', 'xtbopt.xyz', '--hess', '--gfn' + gfn, '--verbose'], cwd='./', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    #stdout, stderr = process.communicate()
                    #output = open("./grain_frequencies.out", "w")
                    #print(stdout.decode(errors="replace"), file=output)
                    #print(stderr.decode(errors="replace"), file=output)   
                    #output.close()
                if os.path.isfile("./grain/xtbhess.xyz"):
                    print(r'Can\'t perform frequencies computation, the grain structure has imaginary frequencies.')
                    exit()
                grain_freq = True
                break
    #Compute the frequencies for the othermethod method. 
    if othermethod is True:
        for i in range(len_grid):
            if i < restart or i < continue_grid:
                continue
            if i < list_range[0] or i > list_range[1]:
                continue 
            if i in list_not_discarded:
                if os.path.isdir('./' + str(i) + '/') is False:
                    print('Folder with geometry needed not found')
                    exit()
                start_GFN_freq(gfn, 'xtbopt.xyz', "BE_" + str(i) + "_frequencies.out", str(i), 0)
                
                #process = subprocess.Popen(['xtb', 'xtbopt.xyz', '--hess', '--gfn' + gfn, '--verbose'], cwd='./' + str(i) + '/', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                #stdout, stderr = process.communicate()
                #output = open("./" + str(i) + "/BE_" + str(i) + "_frequencies.out", "w")
                #print(stdout.decode(errors="replace"), file=output)
                #print(stderr.decode(errors="replace"), file=output)   
                #output.close()

                Results = open("./results_extreme_frequencies_othermethod.txt", "a")
                if os.path.isfile("./" + str(i) + "/xtbhess.xyz"):
                    print(str(i) + " N", file=Results)
                    Results.close()

                    file_list_discarded = open(list_discarded, "a")
                    print(str(i) + " N", file=file_list_discarded)
                    file_list_discarded.close()
                else:
                    print(str(i) + " B", file=Results)
                    Results.close()
    #start of the computation of frequencies for the unfixed-radius part.              
    else:
        file_unfixed_discarded = np.loadtxt("./results_lorenzo_unfixed_grain.txt", dtype=str)
        list_unfixed_discarded = np.atleast_2d(file_unfixed_discarded) #List of BE that are discarded for frequencies computation.
        for i in range(len_grid):
            if i < restart or i < continue_grid:
                continue
            if i < list_range[0] or i > list_range[1]:
                continue 
            if i in list_not_discarded:
                if list_unfixed_discarded[np.where(list_unfixed_discarded[:,0].astype(int) == i)[0],1] == "Y":
                    if os.path.isdir('./' + str(i) + '/unfixed-radius') is False:
                        print('Folder with geometry needed not found')
                        exit()
                    start_GFN_freq(gfn, 'xtbopt.xyz', "BE_" + str(i) + "_frequencies.out", str(i) + '/unfixed-radius', 'xtb.inp')
                    
                    #process = subprocess.Popen(['xtb', '--input', 'xtb.inp', 'xtbopt.xyz', '--hess', '--gfn' + gfn, '--verbose'], cwd='./' + str(i) + '/unfixed-radius', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    #stdout, stderr = process.communicate()
                    #output = open("./" + str(i) + "/unfixed-radius/BE_" + str(i) + "_frequencies.out", "w")
                    #print(stdout.decode(errors="replace"), file=output)
                    #print(stderr.decode(errors="replace"), file=output)   
                    #output.close()

                    Results = open("./results_extreme_frequencies_lorenzo_unfixed_grain.txt", "a")
                    #If imaginary freqencies are present the BE is added to the discarded BE file and the frequencies of the grain are not computed.
                    if os.path.isfile("./" + str(i) + "/unfixed-radius/xtbhess.xyz"):
                        print(str(i) + " N", file=Results)
                        Results.close()
                                
                        file_list_discarded = open(list_discarded, "a")
                        print(str(i) + " N", file=file_list_discarded)
                        file_list_discarded.close()
                    else:
                        #Computation of the grain frequencies
                        subprocess.call(['mkdir', './' + str(i) + '/unfixed-radius/grain_freq'])
                        subprocess.call(['cp', './' + grain, './' + str(i) + '/unfixed-radius/grain_freq/'])
                        subprocess.call(['cp', './' + str(i) + '/unfixed-radius/xtb.inp', './' + str(i) + '/unfixed-radius/grain_freq/'])
                        
                        start_GFN_freq(gfn, 'xtbopt.xyz', 'grain_frequencies.out', str(i) + '/unfixed-radius/grain_freq', 'xtb.inp')

                        #process = subprocess.Popen(['xtb', '--input', 'xtb.inp', 'xtbopt.xyz', '--hess', '--gfn' + gfn, '--verbose'], cwd='./' + str(i) + '/unfixed-radius/grain_freq', stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                        #stdout, stderr = process.communicate()
                        #output = open("./" + str(i) + "/unfixed-radius/grain_freq/grain_frequencies.out", "w")
                        #print(stdout.decode(errors="replace"), file=output)
                        #print(stderr.decode(errors="replace"), file=output)   
                        #output.close()
                        if os.path.isfile("./" + str(i) + "/unfixed-radius/grain_freq/xtbhess.xyz"):
                            print(str(i) + " N", file=Results)
                            Results.close()

                            file_list_discarded = open(list_discarded, "a")
                            print(str(i) + " N", file=file_list_discarded)
                            file_list_discarded.close()
                        else:
                            print(str(i) + " B", file=Results)
                            Results.close()

                else:
                    Results = open("./results_extreme_frequencies_lorenzo_unfixed_grain.txt", "a")
                    print(str(i) + " N", file=Results)
                    Results.close()
            else:
                Results = open("./results_extreme_frequencies_lorenzo_unfixed_grain.txt", "a")
                print(str(i) + " N", file=Results)
                Results.close()

def othermethod_func(restart, continue_grid, list_range):
    #Create every input for the fixed part
    for i in range(len_grid):
        if restart > 0:
            break
        if i < continue_grid:
            continue
        if i < list_range[0] or i > list_range[1]:
            continue 
        subprocess.call(['mkdir', str(i)])
        io.write('./' + str(i) + '/BE_' + str(i) + '.xyz', sphere + sphere_grid[len_sphere + i*len_molecule:len_sphere + (i+1)*len_molecule])
    
    #Compute the BE with the grain totally fixed
    for i in range(len_grid):
        if i < restart or i < continue_grid:
            continue
        if i < list_range[0] or i > list_range[1]:
            continue 
        
        start_GFN(gfn, 'extreme', 'BE_' + str(i) + '.xyz', 'BE_' + str(i) + '.out', str(i), 0)
        #process = subprocess.Popen(['xtb', 'BE_' + str(i) + '.xyz', '--opt', 'extreme', '--gfn' + gfn, '--verbose'], cwd='./' + str(i), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        #stdout, stderr = process.communicate()
        #output = open("./" + str(i) + "/BE_" + str(i) + ".out", "w")
        #print(stdout.decode(errors="replace"), file=output)
        #print(stderr.decode(errors="replace"), file=output)   
        #output.close()
        subprocess.call(['mv', './' + str(i) + '/xtbopt.log', './' + str(i) + '/movie.xyz'])

def start_GFN(gfn, extreme, input_structure, output, folder, input_inp):
    #GFN_start_time = datetime.now()
    if input_inp != 0:
        process = subprocess.Popen(['xtb', '--input', input_inp, input_structure,  '--opt', extreme , '--gfn' + gfn, '--verbose'], cwd='./' + folder, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        process = subprocess.Popen(['xtb', input_structure, '--opt', extreme , '--gfn' + gfn, '--verbose'], cwd='./' + folder, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = open(folder + "/" + output, "w")
    for line in process.stdout:
        print(line.decode(errors="replace"), end='', file=output)
    stdout, stderr = process.communicate()
    print(stderr.decode(errors="replace"), file=output)
    output.close()
    #print('GFN' + str(gfn), str(datetime.now() - GFN_start_time), folder, file=Time_file)

def start_GFN_freq(gfn, input_structure, output, folder, input_inp):
    if input_inp != 0:
        process = subprocess.Popen(['xtb', '--input', input_inp, input_structure, '--hess', '--gfn' + gfn, '--verbose'], cwd='./'  + folder, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        process = subprocess.Popen(['xtb', input_structure, '--hess', '--gfn' + gfn, '--verbose'], cwd='./'  + folder, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = open("./" + folder + "/" + output, "w")
    for line in process.stdout:
        print(line.decode(errors="replace"), end='', file=output)
    stdout, stderr = process.communicate()
    print(stderr.decode(errors="replace"), file=output)   
    output.close()

def molecule_csv(mol):
    df = pd.read_csv('molecules_reactivity_network.csv', sep='\t')
    atoms = io.read(df.loc[df['species'] == mol, 'pwd_xyz'].values[0]) 
    return atoms

if os.path.isfile('./grain/xtbopt.xyz') == True:
    sphere = io.read('./grain/xtbopt.xyz')
    grain = 'grain/xtbopt.xyz'
else:
    subprocess.call(['mkdir', 'grain'])
    subprocess.call(['cp', grain, './grain/'])
    start_GFN(gfn, 'extreme',grain, 'output','grain/', 0)
    sphere = io.read('./grain/xtbopt.xyz')
    grain = 'grain/xtbopt.xyz'


#sphere = io.read('./' + grain) 

continue_grid = grid_continue(list_args_grid_continue[0], list_args_grid_continue[1])

if nofixed is False and onlyunfixed is False and onlyfreq is False:
    if os.path.isfile('grid.xyz') is False and continue_grid == 0:
        grid_building(sphere,level, continue_grid)
    elif os.path.isfile('grid.xyz') is True and continue_grid != 0:
        grid_building(sphere,level, continue_grid)
if os.path.isfile('grid.xyz') is False:
    print('No grid.xyz file')
    exit()
sphere_grid = io.read('./grid.xyz')

len_sphere = len(sphere)

len_sphere_grid = len(sphere_grid)
len_molecule = len(molecule_csv(molecule_to_sample))
len_grid = int((len_sphere_grid - len_sphere)/len_molecule)

if list_range is None:
    list_range = [0, len_grid]

if othermethod is True and onlyfreq is False:
    othermethod_func(restart, continue_grid, list_range)
    restart = 0

if nofixed is False and onlyunfixed is False and onlyfreq is False and othermethod is False:
    fixed(restart, continue_grid, list_range)
    restart = 0
if onlyfixed is False and onlyfreq is False and othermethod is False and nounfixed is False:
    unfixed(restart, continue_grid, list_range)
    restart = 0

if nofreq is False and onlyfixed is False and onlyunfixed is False:
#Block for frequencies computation
    frequencies(restart,othermethod, continue_grid, list_range)
    restart = 0


def energy_opt(file_opt):
    """Reads an .xyz structure produced by GFN-xTB and return the energy of the structure as written in the file."""
    energy = float([key for key in io.read(file_opt).info][1]) #Takes dictionary given by ase when using .info and return the second "key" which is the energy for the GFN-xTB xyz files
    return energy

def ZPE_freq(file_freq):
    """Takes a GFN-xTB output file of the frequencies calculation and return the zero point energy computed."""
    with open(file_freq , "rt") as myfile:
        output = myfile.read()

    start_string=":: zero point energy"
    end_string="Eh   ::" 
            

    start = output.index(start_string) + len(start_string)
    end = output.index(end_string, start)
    zpe_output = float(output[start:end].strip())
    return zpe_output

def HartreetoKjmol(value_Hartree):
    """Takes an energy value in Hartree and convert it to kJ/mol using the ase values"""
    value = value_Hartree * Hartree * mol/kJ
    return value

#output file 

def output():
    """Produces an output file with every BE computed and the ZPE corrected BE if the frequencies were computed."""
    name_file_results = "results.txt"
    file_results = open(name_file_results, "w")
    energy_grain = energy_opt(grain)
    energy_mol = energy_opt('./' + molecule_to_sample + '_xtb/' + molecule_to_sample + '.xyz')
    ZPE_mol = ZPE_freq('./' + molecule_to_sample + '_xtb/frequencies')

    for i in range(len_grid):
        if os.path.isdir('./' + str(i)) is True:
            if os.path.isdir('./' + str(i) + '/unfixed-radius') is True:
                if os.path.isdir('./' + str(i) + '/unfixed-radius/grain_freq') is True:
                    energy_fixed = energy_opt('./' + str(i) + '/xtbopt.xyz')
                    energy_unfixed = energy_opt('./' + str(i) + '/unfixed-radius/xtbopt.xyz')
                    ZPE = ZPE_freq('./' + str(i) + '/unfixed-radius/BE_' + str(i) + '_frequencies.out')
                    ZPE_grain = ZPE_freq('./' + str(i) + '/unfixed-radius/grain_freq/grain_frequencies.out')
                    BE_fixed = HartreetoKjmol(energy_grain + energy_mol - energy_fixed)
                    BE_unfixed = HartreetoKjmol(energy_grain + energy_mol - energy_unfixed)
                    Delta_ZPE = HartreetoKjmol(ZPE - ZPE_grain - ZPE_mol)
                    BE_ZPE = BE_unfixed - Delta_ZPE
                    print(str("{:{width}d}".format(i, width=3)) + str("{: {width}.{prec}f}".format(BE_fixed, width=25, prec=14)) + str("{: {width}.{prec}f}".format(BE_unfixed, width=25, prec=14)) + str("{: {width}.{prec}f}".format(BE_ZPE, width=25, prec=14)) + str("{: {width}.{prec}f}".format(Delta_ZPE, width=25, prec=14)), file=file_results)
                else:
                    energy_fixed = energy_opt('./' + str(i) + '/xtbopt.xyz')
                    energy_unfixed = energy_opt('./' + str(i) + '/unfixed-radius/xtbopt.xyz')
                    BE_fixed = HartreetoKjmol(energy_grain + energy_mol - energy_fixed)
                    BE_unfixed = HartreetoKjmol(energy_grain + energy_mol - energy_unfixed)
                    print(str("{:{width}d}".format(i, width=3)) + str("{: {width}.{prec}f}".format(BE_fixed, width=25, prec=14)) + str("{: {width}.{prec}f}".format(BE_unfixed, width=25, prec=14)), file=file_results)

            elif os.path.isfile('./' + str(i) + '/xtb.inp') is False:
                energy = energy_opt('./' + str(i) + '/xtbopt.xyz')
                ZPE = ZPE_freq('./' + str(i) + '/BE_' + str(i) + '_frequencies.out')
                BE = HartreetoKjmol(energy_grain + energy_mol - energy)
                ZPE_grain = ZPE_freq('./grain/grain_frequencies.out')
                Delta_ZPE = HartreetoKjmol(ZPE - ZPE_grain - ZPE_mol)
                BE_ZPE = BE - Delta_ZPE
                print(str("{:{width}d}".format(i, width=3)) + str("{: {width}.{prec}f}".format(BE, width=25, prec=14)) + str("{: {width}.{prec}f}".format(BE_ZPE, width=25, prec=14)) + str("{: {width}.{prec}f}".format(Delta_ZPE, width=25, prec=14)), file=file_results)
            else:
                energy_fixed = energy_opt('./' + str(i) + '/xtbopt.xyz')
                BE_fixed = HartreetoKjmol(energy_grain + energy_mol - energy_fixed)
                print(str("{:{width}d}".format(i, width=3)) + str("{: {width}.{prec}f}".format(BE_fixed, width=25, prec=14)), file=file_results)
        #else:
            #print('No results to print in the output file.')
            #exit()
    file_results.close()

output()