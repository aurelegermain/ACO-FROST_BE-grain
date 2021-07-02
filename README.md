# BE-grain

A python code using GFN-xTB [[1][2][3]](#1) to compute binding energies of the chosen molecule around grain structure provided by the user.

## Arguments

- `grain_structure_name.xyz` XYZ file containing the grain structure **Mandatory**
- `-level int` the grid level wanted by the user. Level 0 has 12 grid points, 1 has 42, 3 has 162. Default = 0.
- `-mol str` Molecule to study. Example: H2O. **Mandatory**
- `-g str ot int` GFN-xTB method to use (0,1,2 or ff). Default = 2.
- `-r float` Radius size in Ångström to use during the unfixed computation of the BEs. Default = 5.
- `-restart int` Indicates for the calculation is a restart. The BE to restart to needs to be specified. Default = 0.
- `-gc int int` After a first calculation if the user wants the grid to be increased. Takes two arguments, the previous grid level and the new grid level desired. Default = 0 0.
- `-onlyfixed` Only the computation with the grain entirely fixed is done.
- `-onlyunfixed` Only the conutation with the unfixed radius is done. **Needs the fixed files.** Argument to use to continue a previous computation using ` -onlyfixed`.
- `-onlyfreq` Only frequencies computations. **Needs the fixed and unfixed files.** Argument to use to continue a previous computation.
- `-nofreq` No frequencies computation done.
- `-nofixed` No computation with the entire grain fixed done. **Needs the fixed files**. Argument to use to ontinue a previous computation.
- `-om` BE computation without fixing anything.

## References
<a id ="1">[1]</a>
Bannwarth, C.; Ehlert, S.; Grimme, S. GFN2-xTB—An Accurate and Broadly Parametrized Self-Consistent Tight-Binding Quantum Chemical Method with Multipole Electrostatics and Density-Dependent Dispersion Contributions J. Chem. Theory Comput. 2019, 15 (3), 1652–1671 DOI: 10.1021/acs.jctc.8b01176

<a id ="2">[2]</a>
Grimme, S.; Bannwarth, C.; Shushkov, P. A Robust and Accurate Tight-Binding Quantum Chemical Method for Structures, Vibrational Frequencies, and Noncovalent Interactions of Large Molecular Systems Parameterized for All spd-Block Elements (Z = 1-86). J. Chem. Theory Comput. 2017, 13 (5), 1989-2009 DOI: 10.1021/acs.jctc.7b00118

<a id ="3">[3]</a>
Spicher, S.; Grimme, S. Robust atomistic modeling of materials, organometallic and biochemical systems Angew. Chem. Int. Ed. 2020, accepted article, DOI: 10.1002/anie.202004239