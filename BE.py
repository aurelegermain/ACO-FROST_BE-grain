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
from ase import io, Atoms, neighborlist
from ase.units import kJ,Hartree,mol
from ase.build import molecule
import argparse
import math
import os
from itertools import combinations
import conflictsparse

rng = np.random.default_rng()

parser = argparse.ArgumentParser()
parser.add_argument("grain", metavar="G", help="Grain model to sample in .xyz", type=str)
parser.add_argument("-level", "--level", help="Grid level needed", type=int, default="0")
parser.add_argument("-mol", "--molecule", help="molecule to sample", type=str, default="0", required=True)
parser.add_argument("-g", "--gfn", help="GFN-xTB method to use (0,1,2, or ff)", default="2")
parser.add_argument("-r", "--radius", help="Radius for unfixing molecules", type=float, default="5")
parser.add_argument("-om","--othermethod", action='store_true', help="Other method without the fixing of anything. Temporary name.")
parser.add_argument("-restart", "--restart", help="If the calculation is a restart and from where it restart", type=int, default="0")
parser.add_argument("-gc", "--grid_continue", nargs='+', help="If you want to increase the size of the grid. First number is the first level of grid used, second number is the level desired", type=int, default=[0,0])

#Conflicting options part

#conflict between -onlyfreq and -nofreq
group_freq = parser.add_mutually_exclusive_group()
group_freq.add_argument('-nofreq', action='store_true', help="No frequencies computation")
only_freq_arg = group_freq.add_argument('-onlyfreq', action='store_true', help="Only the frequencies computation. Needs the fixed or unfixed files")

#conflict between -onlyfixed and -nofixed
group_fixed = parser.add_mutually_exclusive_group()
group_fixed.add_argument('-nofixed', action='store_true', help="The grain is totally unfixed")
only_fixed_arg = group_fixed.add_argument('-onlyfixed', action='store_true', help="Only the opt of the fixed grain")

#conflict between -onlyfixed -onlyunfixed and -onlyfreq
group_only = parser.add_mutually_exclusive_group()
group_only.add_argument('-onlyunfixed', action='store_true', help="Only the opt of the unfixed grain. Needs the fixed files")
group_only._group_actions.append(only_freq_arg)
group_only._group_actions.append(only_fixed_arg)

args = parser.parse_args()

level = args.level
grain = args.grain
molecule_to_sample = args.molecule.upper() #The molecule indicated by the user is automatically put in upper case 
gfn = str(args.gfn)
radius_unfixed = args.radius
nofreq =args.nofreq
nofixed = args.nofixed
onlyfixed = args.onlyfixed
onlyunfixed = args.onlyunfixed
onlyfreq = args.onlyfreq
othermethod = args.othermethod
restart = args.restart
list_args_grid_continue = args.grid_continue

#parameters needed for the starting positions. 
#Should be put inside the function at some point (not used anywhere else I think)
distance = 2.5
coeff_min = 1.00
coeff_max = 1.2
steps = 0.1

#This is the file that contains the id of the BE that had opt problems in fixed, or are too far from the grain in unfixed
list_discarded = "list_discarded.txt"

def grid_continue(starting_grid, ending_grid):
    """ 
    If the user wants to increase the number of BEs by increasing the size of the grid. 
    Takes the level of the old grid and the level of the desired one and return the number of BE to ignore so that as to not compute the old grid again
    """
    list_size_grid = np.atleast_1d(np.zeros(ending_grid + 1))

    for i in range(len(list_size_grid)):
        if i == 0:
            list_size_grid[i] = 12
        else:
            list_size_grid[i] = list_size_grid[i - 1] + 2**(2*(i + 1) + 1) - 2**(2*(i - 1) + 1) #this is the equation that gives the number of grid points for the level n (depends on N(n-1))
    if (ending_grid - starting_grid) ==0:
        continue_grid = 0
    else:
        continue_grid = int(list_size_grid[starting_grid])

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
    
    #this is simply to project on a sphere with a radius taken from the grain model to sample
    distance_max = np.amax(distances_3d(sphere)) + 1

    #Build an ase Atoms object using the studied molecule. Then perform an xtb opt and a frequency calculation. The opt geometry will be used for the input file, and the frequency calculation will be used for ZPE corrected BE 
    if continue_grid == 0: #If we continue from a previsous grid this part was already done
        #make a directory with the name of the molecule + _xtb. To store the xtb files of the molecule to sample
        subprocess.call(['mkdir', molecule_to_sample + '_xtb'])

        io.write('./' + molecule_to_sample + '_xtb/' + molecule_to_sample + '_inp.xyz', molecule(molecule_to_sample))
        process = subprocess.Popen(['xtb', molecule_to_sample + '_inp.xyz', '--opt', 'extreme', '--gfn' + gfn, '--verbose'], cwd='./' + molecule_to_sample + '_xtb', stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        stdout, stderr = process.communicate()
        output = open("./" + molecule_to_sample + "_xtb/output", "w")
        print(stdout.decode(), file=output)
        print(stderr.decode(), file=output)   
        output.close()

        subprocess.call(['mv', './' + molecule_to_sample + '_xtb/xtbopt.xyz', './' + molecule_to_sample + '_xtb/' + molecule_to_sample + '.xyz'])

        process = subprocess.Popen(['xtb', molecule_to_sample + '.xyz', '--hess', '--gfn' + gfn, '--verbose'], cwd='./' + molecule_to_sample + '_xtb', stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        stdout, stderr = process.communicate()
        output = open("./" + molecule_to_sample + "_xtb/frequencies", "w")
        print(stdout.decode(), file=output)
        print(stderr.decode(), file=output)   
        output.close()

    #this is the molecule that will be added for each rid point.
    atoms_to_add = io.read('./' + molecule_to_sample + '_xtb/' + molecule_to_sample + '.xyz') 

    #replace each grid point by the molecule studied and apply a random orientation to it
    #The grid radius is also changed and set to the "distance_max" variable
    for i in range(len(grid) - continue_grid):
        atoms_to_add2 = deepcopy(atoms_to_add)
        #rotate the molecule randomly
        angle_mol = rng.random(3)*360
        atoms_to_add2.rotate(angle_mol[0], "x")
        atoms_to_add2.rotate(angle_mol[1], "y")
        atoms_to_add2.rotate(angle_mol[2], "z")

        for j in range(len(atoms_to_add)):
            atoms_to_add2[j].position = atoms_to_add2[j].position + grid[i + continue_grid]*distance_max
        if i == 0:
            #atoms is the ase atoms object containing all the posistions of the studied molecule 
            atoms = atoms_to_add2
        else:
            atoms = atoms + atoms_to_add2
    if continue_grid == 0:
        #write the file containing all the positions with the grain structure used
        io.write('./grid_first.xyz', sphere + atoms)
    else:
        #if the computation continues from a previous one we update the file
        grid_first_old = io.read('./grid_first.xyz') 
        io.write('./grid_first.xyz', grid_first_old + atoms)

    position_mol = np.zeros(3)
    for i in range(len(grid) - continue_grid):
        #Maybe a too complicated method
        #Compute the minimum distance between the molecule 
        iter_steps = 0
        i_continue = int(i + continue_grid)
        radius = np.sqrt(np.square(grid[i_continue,0]*distance_max) + np.square(grid[i_continue,1]*distance_max) + np.square(grid[i_continue,2]*distance_max))
        theta = np.arccos(grid[i_continue,2]*distance_max/radius)
        phi = np.arctan2(grid[i_continue,1]*distance_max,grid[i_continue,0]*distance_max)
        molecule_to_move = barycentre(atoms[i*len(atoms_to_add):(i+1)*len(atoms_to_add)])
        while np.amin(distances_ab(atoms[i*len(atoms_to_add):(i+1)*len(atoms_to_add)], sphere)) > distance * coeff_max or np.amin(distances_ab(atoms[i*len(atoms_to_add):(i+1)*len(atoms_to_add)], sphere)) < distance / coeff_min:
            
            if np.amin(distances_ab(atoms[i*len(atoms_to_add):(i+1)*len(atoms_to_add)], sphere)) > distance * coeff_max:
                iter_steps += - 1
            else:
                iter_steps += 1
            #change the distance between the molecule and the grain by increments of 0.1A 
            position_mol[0] = (radius + iter_steps*steps)*np.sin(theta)*np.cos(phi)
            position_mol[1] = (radius + iter_steps*steps)*np.sin(theta)*np.sin(phi)
            position_mol[2] = (radius + iter_steps*steps)*np.cos(theta)

            for j in range(len(atoms_to_add)):
                atoms[i*len(atoms_to_add)+j].position = molecule_to_move[j].position + position_mol
    if continue_grid == 0: 
        #writ the file with all the final input positions
        io.write('./grid.xyz', sphere + atoms)
    else:
        #if the computation continues from a previous one we update the file
        grid_old = io.read('./grid.xyz') 
        io.write('./grid.xyz', grid_old + atoms)

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
    cutOff = neighborlist.natural_cutoffs(mol)
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
def fixed(restart, continue_grid):
    """
    Read the grid.xyz file and produce separate input files for each position. 
    Then compute the GFN-xTB opt of the inputs and use an xtb.inp file to fix the grain.
    """
    #Create every input for the fixed part
    for i in range(len_grid):
        if i < continue_grid:
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
        process = subprocess.Popen(['xtb', '--input', 'xtb.inp', 'BE_' + str(i) + '.xyz', '--opt', 'extreme', '--gfn' + gfn, '--verbose'], cwd='./' + str(i), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        stdout, stderr = process.communicate()
        output = open("./" + str(i) + "/BE_" + str(i) + ".out", "w")
        print(stdout.decode(), file=output)
        print(stderr.decode(), file=output)   
        output.close()
        subprocess.call(['mv', './' + str(i) + '/xtbopt.log', './' + str(i) + '/movie.xyz'])

        #if the computation has not converged the BE is added to a file indicating every BE that did not work and should not be used for further computations.
        if os.path.isfile("./" + str(i) + "/NOT_CONVERGED"):
            file_list_discarded = open(list_discarded, "a")
            print(str(i) + "    N", file=file_list_discarded)
            file_list_discarded.close()

def unfixed(restart, continue_grid):
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
        print(list_fix)
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
        if i in list_not_discarded:
            process = subprocess.Popen(['xtb', '--input', 'xtb.inp', 'BE_' + str(i) + '.xyz', '--opt', 'extreme', '--gfn' + gfn, '--verbose'], cwd='./' + str(i) + '/unfixed-radius', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
            stdout, stderr = process.communicate()
            output = open("./" + str(i) + "/unfixed-radius/BE_" + str(i) + ".out", "w")
            print(stdout.decode(), file=output)
            print(stderr.decode(), file=output)   
            output.close()
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
                    print(list_fix)
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
    
    
                    process = subprocess.Popen(['xtb', '--input', 'xtb.inp', 'BE_' + str(i) + '.xyz', '--opt', 'extreme', '--gfn' + gfn, '--verbose'], cwd='./' + str(i) + '/unfixed-radius', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
                    stdout, stderr = process.communicate()
                    output = open("./" + str(i) + "/unfixed-radius/BE_" + str(i) + ".out", "w")
                    print(stdout.decode(), file=output)
                    print(stderr.decode(), file=output)   
                    output.close()
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

def frequencies(restart, othermethod, continue_grid):
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
        if i in list_not_discarded:
            if os.path.isdir('./' + str(i) + '/') is True and os.path.isfile('./' + str(i) + '/xtb.inp') is False and grain_freq is False:
                othermethod = True
                if continue_grid == 0 or restart == 0:
                    process = subprocess.Popen(['xtb', grain, '--hess', '--gfn' + gfn, '--verbose'], cwd='./', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout, stderr = process.communicate()
                    output = open("./grain_frequencies.out", "w")
                    print(stdout.decode(), file=output)
                    print(stderr.decode(), file=output)   
                    output.close()
                if os.path.isfile("./xtbhess.xyz"):
                    print(r'Can\'t perform frequencies computation, the grain structure has imaginary frequencies.')
                    exit()
                grain_freq = True
                break
    #Compute the frequencies for the othermethod method. 
    if othermethod is True:
        for i in range(len_grid):
            if i < restart or i < continue_grid:
                continue
            if i in list_not_discarded:
                if os.path.isdir('./' + str(i) + '/') is False:
                    print('Folder with geometry needed not found')
                    exit()
                process = subprocess.Popen(['xtb', 'xtbopt.xyz', '--hess', '--gfn' + gfn, '--verbose'], cwd='./' + str(i) + '/', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                output = open("./" + str(i) + "/BE_" + str(i) + "_frequencies.out", "w")
                print(stdout.decode(), file=output)
                print(stderr.decode(), file=output)   
                output.close()

                Results = open("./results_extreme_frequencies_othermethod.txt", "a")
                if os.path.isfile("./" + str(i) + "/xtbhess.xyz"):
                    print("N", file=Results)
                    Results.close()

                    file_list_discarded = open(list_discarded, "a")
                    print(str(i) + "    N", file=file_list_discarded)
                    file_list_discarded.close()
                else:
                    print("B", file=Results)
                    Results.close()
    #start of the computation of frequencies for the unfixed-radius part.              
    else:
        file_unfixed_discarded = np.loadtxt("./results_lorenzo_unfixed_grain.txt", dtype=str)
        list_unfixed_discarded = np.atleast_2d(file_unfixed_discarded) #List of BE that are discarded for frequencies computation.
        for i in range(len_grid):
            if i < restart or i < continue_grid:
                continue
            if i in list_not_discarded:
                if list_unfixed_discarded[i,1] == "Y":
                    if os.path.isdir('./' + str(i) + '/unfixed-radius') is False:
                        print('Folder with geometry needed not found')
                        exit()
                    process = subprocess.Popen(['xtb', '--input', 'xtb.inp', 'xtbopt.xyz', '--hess', '--gfn' + gfn, '--verbose'], cwd='./' + str(i) + '/unfixed-radius', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout, stderr = process.communicate()
                    output = open("./" + str(i) + "/unfixed-radius/BE_" + str(i) + "_frequencies.out", "w")
                    print(stdout.decode(), file=output)
                    print(stderr.decode(), file=output)   
                    output.close()

                    Results = open("./results_extreme_frequencies_lorenzo_unfixed_grain.txt", "a")
                    #If imaginary freqencies are present the BE is added to the discarded BE file and the frequencies of the grain are not computed.
                    if os.path.isfile("./" + str(i) + "/unfixed-radius/xtbhess.xyz"):
                        print("N", file=Results)
                        Results.close()
                                
                        file_list_discarded = open(list_discarded, "a")
                        print(str(i) + "    N", file=file_list_discarded)
                        file_list_discarded.close()
                    else:
                        #Computation of the grain frequencies
                        subprocess.call(['mkdir', './' + str(i) + '/unfixed-radius/grain_freq'])
                        subprocess.call(['cp', './' + grain, './' + str(i) + '/unfixed-radius/grain_freq/'])
                        subprocess.call(['cp', './' + str(i) + '/unfixed-radius/xtb.inp', './' + str(i) + '/unfixed-radius/grain_freq/'])
                        process = subprocess.Popen(['xtb', '--input', 'xtb.inp', grain, '--hess', '--gfn' + gfn, '--verbose'], cwd='./' + str(i) + '/unfixed-radius/grain_freq', stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                        stdout, stderr = process.communicate()
                        output = open("./" + str(i) + "/unfixed-radius/grain_freq/grain_frequencies.out", "w")
                        print(stdout.decode(), file=output)
                        print(stderr.decode(), file=output)   
                        output.close()
                        if os.path.isfile("./" + str(i) + "/unfixed-radius/grain_freq/xtbhess.xyz"):
                            print("N", file=Results)
                            Results.close()

                            file_list_discarded = open(list_discarded, "a")
                            print(str(i) + "    N", file=file_list_discarded)
                            file_list_discarded.close()
                        else:
                            print("B", file=Results)
                            Results.close()

                else:
                    Results = open("./results_extreme_frequencies_lorenzo_unfixed_grain.txt", "a")
                    print(str(i) + " N", file=Results)
                    Results.close()
            else:
                Results = open("./results_extreme_frequencies_lorenzo_unfixed_grain.txt", "a")
                print(str(i) + " N", file=Results)
                Results.close()

def othermethod_func(restart, continue_grid):
    #Create every input for the fixed part
    for i in range(len_grid):
        if restart > 0:
            break
        if i < continue_grid:
            continue
        subprocess.call(['mkdir', str(i)])
        io.write('./' + str(i) + '/BE_' + str(i) + '.xyz', sphere + sphere_grid[len_sphere + i*len_molecule:len_sphere + (i+1)*len_molecule])
    
    #Compute the BE with the grain totally fixed
    for i in range(len_grid):
        if i < restart or i < continue_grid:
            continue
        process = subprocess.Popen(['xtb', 'BE_' + str(i) + '.xyz', '--opt', 'extreme', '--gfn' + gfn, '--verbose'], cwd='./' + str(i), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        stdout, stderr = process.communicate()
        output = open("./" + str(i) + "/BE_" + str(i) + ".out", "w")
        print(stdout.decode(), file=output)
        print(stderr.decode(), file=output)   
        output.close()
        subprocess.call(['mv', './' + str(i) + '/xtbopt.log', './' + str(i) + '/movie.xyz'])

sphere = io.read('./' + grain) 

continue_grid = grid_continue(list_args_grid_continue[0], list_args_grid_continue[1])

if nofixed is False and onlyunfixed is False and onlyfreq is False:
    grid_building(sphere,level, continue_grid)
if os.path.isfile('grid.xyz') is False:
    print('No grid.xyz file')
    exit()
sphere_grid = io.read('./grid.xyz')

len_sphere = len(sphere)

len_sphere_grid = len(sphere_grid)
len_molecule = len(molecule(molecule_to_sample))
len_grid = int((len_sphere_grid - len_sphere)/len_molecule)

if othermethod is True and onlyfreq is False:
    othermethod_func(restart, continue_grid)
    restart = 0

if nofixed is False and onlyunfixed is False and onlyfreq is False and othermethod is False:
    fixed(restart, continue_grid)
    restart = 0
if onlyfixed is False and onlyfreq is False and othermethod is False:
    unfixed(restart, continue_grid)
    restart = 0

if nofreq is False and onlyfixed is False and onlyunfixed is False:
#Block for frequencies computation
    frequencies(restart,othermethod, continue_grid)
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
                    energy_unfixed = energy_opt('./' + str(i) + '/xtbopt.xyz')
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
                ZPE_grain = ZPE_freq('./grain_frequencies.out')
                Delta_ZPE = HartreetoKjmol(ZPE - ZPE_grain - ZPE_mol)
                BE_ZPE = BE - Delta_ZPE
                print(str("{:{width}d}".format(i, width=3)) + str("{: {width}.{prec}f}".format(BE, width=25, prec=14)) + str("{: {width}.{prec}f}".format(BE_ZPE, width=25, prec=14)) + str("{: {width}.{prec}f}".format(Delta_ZPE, width=25, prec=14)), file=file_results)
            else:
                energy_fixed = energy_opt('./' + str(i) + '/xtbopt.xyz')
                BE_fixed = HartreetoKjmol(energy_grain + energy_mol - energy_fixed)
                print(str("{:{width}d}".format(i, width=3)) + str("{: {width}.{prec}f}".format(BE_fixed, width=25, prec=14)), file=file_results)
        else:
            print('No results to print in the output file.')
            exit()
    file_results.close()

output()