# %%
import pandas as pd
from copy import deepcopy
import subprocess
import numpy as np
from scipy.spatial import ConvexHull
from ase import io, Atoms, neighborlist
from ase.build import molecule
import argparse
import math
import os
from itertools import combinations

rng = np.random.default_rng()

parser = argparse.ArgumentParser()
parser.add_argument("grain", metavar="G", help="Grain model to sample", type=str)
parser.add_argument("-level", "--level", help="Grid level needed", type=int, default="0")
parser.add_argument("-mol", "--molecule", help="molecule to sample", type=str, default="0", required=True)
parser.add_argument("-g", "--gfn", help="GFN-xTB method to use", default="2")
parser.add_argument("-r", "--radius", help="Radius for unfixing molecules", type=float, default="5")

args = parser.parse_args()

level = args.level
grain = args.grain
molecule_to_sample = args.molecule.upper()
gfn = str(args.gfn)
radius_unfixed = args.radius


distance = 2.5
coeff_min = 1.00
coeff_max = 1.2
steps = 0.1

list_discarded = "list_discarded.txt"

def grid_building(sphere, level):
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
    
        #build the list of each point to be added to constructuct the next level grid
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

    #make a directory wth the name of the molecule + _xtb. To store the xtb files of the molecule to sample
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

 
    for i in range(len(grid)):
        atoms_to_add2 = deepcopy(atoms_to_add)
        #rotate the molecule randomly
        angle_mol = rng.random(3)*360
        atoms_to_add2.rotate(angle_mol[0], "x")
        atoms_to_add2.rotate(angle_mol[1], "y")
        atoms_to_add2.rotate(angle_mol[2], "z")

        for j in range(len(atoms_to_add)):
            atoms_to_add2[j].position = atoms_to_add2[j].position + grid[i]*distance_max
        if i == 0:
            atoms = atoms_to_add2
        else:
            atoms = atoms + atoms_to_add2
    io.write('./grid_first.xyz', sphere + atoms)

    position_mol = np.zeros(3)
    for i in range(len(grid)): #atoms[i,len(atom2) + 1]
        iter_steps = 0
        radius = np.sqrt(np.square(grid[i,0]*distance_max) + np.square(grid[i,1]*distance_max) + np.square(grid[i,2]*distance_max))
        theta = np.arccos(grid[i,2]*distance_max/radius)
        phi = np.arctan2(grid[i,1]*distance_max,grid[i,0]*distance_max)
        molecule_to_move = barycentre(atoms[i*len(atoms_to_add):(i+1)*len(atoms_to_add)])
        while np.amin(distances_ab(atoms[i*len(atoms_to_add):(i+1)*len(atoms_to_add)], sphere)) > distance * coeff_max or np.amin(distances_ab(atoms[i*len(atoms_to_add):(i+1)*len(atoms_to_add)], sphere)) < distance / coeff_min:
            
            if np.amin(distances_ab(atoms[i*len(atoms_to_add):(i+1)*len(atoms_to_add)], sphere)) > distance * coeff_max:
                iter_steps += - 1
            else:
                iter_steps += 1
            
            position_mol[0] = (radius + iter_steps*steps)*np.sin(theta)*np.cos(phi)
            position_mol[1] = (radius + iter_steps*steps)*np.sin(theta)*np.sin(phi)
            position_mol[2] = (radius + iter_steps*steps)*np.cos(theta)

            for j in range(len(atoms_to_add)):
                atoms[i*len(atoms_to_add)+j].position = molecule_to_move[j].position + position_mol
   
    io.write('./grid.xyz', sphere + atoms)

def distances_3d(atoms):
    x = atoms.get_positions()[:,0] if hasattr(atoms, '__len__') else atoms.position[0]
    y = atoms.get_positions()[:,1] if hasattr(atoms, '__len__') else atoms.position[1]
    z = atoms.get_positions()[:,2] if hasattr(atoms, '__len__') else atoms.position[2]
    list_distances = np.sqrt(x**2 + y**2 + z**2)
    return list_distances

def distances_ab(atoms_a, atoms_b):
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
    positions = atoms.get_positions()
    barycentre_position = np.array([np.sum(positions[:,0])/len(positions[:,0]),np.sum(positions[:,1])/len(positions[:,0]),np.sum(positions[:,2])/len(positions[:,0])])
    new_positions = np.zeros(np.shape(positions))
    for i in range(len(new_positions)):
        new_positions[i,:] = positions[i,:] - barycentre_position
    atoms.set_positions(new_positions)
    return atoms

def FromXYZtoDataframeMolecule(input_file):
	#read and encode .xyz fiel into Pandas Dataframe
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

sphere = io.read('./' + grain) 
print(grid_building(sphere,level)) 
sphere_grid = io.read('./grid.xyz')

len_sphere = len(sphere)

len_sphere_grid = len(sphere_grid)
len_molecule = len(molecule(molecule_to_sample))
len_grid = int((len_sphere_grid - len_sphere)/len_molecule)

#Start of the grain totally fixed part of BE computation

#Create every input for the fixed part
for i in range(len_grid):
    subprocess.call(['mkdir', str(i)])
    file_xtb_input = open("./" + str(i) + "/xtb.inp","w")
    print("$constrain", file=file_xtb_input)
    print("    atoms: 1-" + str(len_sphere), file=file_xtb_input)
    print("$end", file=file_xtb_input)
    file_xtb_input.close()
    io.write('./' + str(i) + '/BE_' + str(i) + '.xyz', sphere + sphere_grid[len_sphere + i*len_molecule:len_sphere + (i+1)*len_molecule])

#Compute the BE with the grain totally fixed
for i in range(len_grid):
    process = subprocess.Popen(['xtb', '--input', 'xtb.inp', 'BE_' + str(i) + '.xyz', '--opt', 'extreme', '--gfn' + gfn, '--verbose'], cwd='./' + str(i), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    stdout, stderr = process.communicate()
    output = open("./" + str(i) + "/BE_" + str(i) + ".out", "w")
    print(stdout.decode(), file=output)
    print(stderr.decode(), file=output)   
    output.close()
    subprocess.call(['mv', './' + str(i) + '/xtbopt.log', './' + str(i) + '/movie.xyz'])

#Start of the unfixed radius part of the BE computation
for i in range(len_grid):
    subprocess.call(['mkdir', './' + str(i) + '/unfixed-radius'])
    subprocess.call(['cp', './' + str(i) + '/xtbopt.xyz', './' + str(i) + '/unfixed-radius/BE_' + str(i) + '.xyz'])

    input_BE_unfixed  = './' + str(i) + '/unfixed-radius/BE_' + str(i) + '.xyz'
    df_xyz_test = FromXYZtoDataframeMolecule(input_BE_unfixed) 
    mol_ref = df_xyz_test['Molecules'].max()
    df_xyz_test = LabelMoleculesRadius(df_xyz_test,mol_ref,radius_unfixed)
    grain_structure = df_xyz_test.to_numpy()

    for k in range(len(grain_structure[:,5])):
        if grain_structure[k,5] == "M":
            if "list_fix" in globals():
                list_fix = np.append(list_fix,k)
            else:
                list_fix = k
        elif grain_structure[k,5] == "H" and grain_structure[k,4]!=mol_ref:
            if "list_not_fix" in globals():
                list_not_fix = np.append(list_not_fix,k)
            else:
                list_not_fix = k

    file_xtb_unfixed_input = open("./" + str(i) + "/unfixed-radius/xtb.inp","w")
    print("$constrain", file=file_xtb_unfixed_input)
    print("    atoms: ", end="", file=file_xtb_unfixed_input)

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

    if "list_not_fix" in globals():
        del list_not_fix
    else:
        file_list_discarded = open(list_discarded, "a")
        print(str(i) + "    N", file=file_list_discarded)
        file_list_discarded.close()

    if "list_fix" in globals(): 
        del list_fix

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

for i in range(len_grid):
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

        while equal_structure!=True:
            
            if os.path.isfile("NOT_CONVERGED"):
                equal_structure = True
                Results = open("./results_lorenzo_unfixed_grain.txt", "a")
                print(str(i) + " N", file=Results)
                Results.close()
                continue
            
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
    
            if np.array_equal(grain_structure[:,5],grain_structure2[:,5])==False:
                print(str(i) + " prout")
                subprocess.call(['mv', './' + str(i) + '/unfixed-radius', './' + str(i) + '/' + str(restart_BE)])
                subprocess.call(['mkdir', './' + str(i) + '/unfixed-radius'])
                subprocess.call(['cp', './' + str(i) + '/' + str(restart_BE) + '/xtbopt.xyz', './' + str(i) + '/unfixed-radius/BE_' + str(i) + '.xyz'])
    
                for k in range(len(grain_structure[:,5])):
                    if grain_structure[k,5] == "M":
                        if "list_fix" in globals():
                           list_fix = np.append(list_fix,k)
                        else:
                             list_fix = k
                    elif grain_structure[k,5] == "H" and grain_structure[k,4]!=mol_ref:
                        if "list_not_fix" in globals():
                            list_not_fix = np.append(list_not_fix,k)
                        else:
                            list_not_fix = k
    
                file_xtb_unfixed_input = open("./" + str(i) + "/unfixed-radius/xtb.inp","w")
                print("$constrain", file=file_xtb_unfixed_input)
                print("    atoms: ", end="", file=file_xtb_unfixed_input)
            
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
                subprocess.call(['mv', './' + str(i) + '/' + str(l), './' + str(i) + '/unfixed-radius'])
        if os.path.isfile("NOT_CONVERGED") == False:
            Results = open("./results_lorenzo_unfixed_grain.txt", "a")
            print(str(i) + " Y", file=Results)
            Results.close()

file_unfixed_discarded = np.loadtxt("./results_lorenzo_unfixed_grain.txt", dtype=str)
list_unfixed_discarded = np.atleast_2d(file_unfixed_discarded)

for i in range(len_grid):
    if i in list_not_discarded:
        if list_unfixed_discarded[i,1] == "Y":
                process = subprocess.Popen(['xtb', '--input', 'xtb.inp', 'xtbopt.xyz', '--hess', '--gfn' + gfn, '--verbose'], cwd='./' + str(i) + '/unfixed-radius', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                output = open("./" + str(i) + "/unfixed-radius/BE_" + str(i) + "_frequencies.out", "w")
                print(stdout.decode(), file=output)
                print(stderr.decode(), file=output)   
                output.close()

                Results = open("./results_extreme_frequencies_lorenzo_unfixed_grain.txt", "a")
                if os.path.isfile("./" + str(i) + "/unfixed-radius/xtbhess.xyz"):
                    print("N", file=Results)
                    Results.close()
                else:

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