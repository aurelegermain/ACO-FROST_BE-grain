# BE-grain

A python code using GFN-xTB [[1][2][3]](#1) to compute binding energies of the chosen molecule around grain structure provided by the user.

## Arguments

- `grain_structure_name.xyz` XYZ file containing the grain structure **Mandatory**
- `-level int` the grid level wanted by the user. Level 0 has 12 grid points, 1 has 42, 3 has 162 (From [[4]](#4)). *Default = 0.*
- `-mol str` Molecule to study. Example: H2O. **Mandatory**
- `-g str or int` GFN-xTB method to use (0,1,2 or ff). *Default = 2.*
- `-r float` Radius size in Ångström to use during the unfixed computation of the BEs. *Default = 5.*
- `-restart int` Indicates for the calculation is a restart. The BE to restart to needs to be specified. *Default = 0.*
- `-gc int int` After a first calculation if the user wants the grid to be increased. Takes two arguments, the previous grid level and the new grid level desired. *Default = 0 0.*
- `-onlyfixed` Only the computation with the grain entirely fixed is done. *Default = False.*
- `-onlyunfixed` Only the conutation with the unfixed radius is done. **Needs the fixed files.** Argument to use to continue a previous computation using ` -onlyfixed`. *Default = False.*
- `-onlyfreq` Only frequencies computations. **Needs the fixed and unfixed files.** Argument to use to continue a previous computation. *Default = False.*
- `-nofreq` No frequencies computation done. *Default = False.*
- `-nofixed` No computation with the entire grain fixed done. **Needs the fixed files**. Argument to use to ontinue a previous computation. *Default = False.*
- `-om` BE computation without fixing anything. *Default = False.*

## Examples

- `python3 BE.py sphere.xyz -level 1 -mol H2O -g ff` Compute the BEs of Water (fixed, unfixed, and ZPE corrected)around a grain structure contained in *sphere.xyz* using a level 1 grid (42 positions) and *GFN-FF*.
- `python3 BE.py grain.xyz -level 2 -mol NH3 -g 2 -r 4 -nofreq` Compute the BEs of Ammonia (fixed and unfixed) around a grain structure contained in *grain.xyz* using a level 2 grid (162 posisitions), *GFN2*, and a radius of 4Å for the unfixed computation. Due to `-nofreq`, no frequency computaton are done.
- `python3 BE.py grain.xyz -level 2 -mol CH3 -g 1 -gc 0 2 -onlyfixed` Takes a previous calculation done using a level 0 grid and increase to a level 2 computing only the fixed BEs using *GFN0*. 

- `python3 BE.py grain.xyz -level 2 -mol HCOOH -g 1 -restart 87 -nofixed` Takes a previously stoped computation and restart it at the BE in the folder 87. In that case we presume that the fixed computation were done but the unfixed stopped at the BE 87. The computation will continue for the unfixed BE and then compute the frequencies starting from the BE in the folder 0. 

## References
<a id ="1">[1]</a>
Bannwarth, C.; Ehlert, S.; Grimme, S. GFN2-xTB—An Accurate and Broadly Parametrized Self-Consistent Tight-Binding Quantum Chemical Method with Multipole Electrostatics and Density-Dependent Dispersion Contributions J. Chem. Theory Comput. **2019**, 15 (3), 1652–1671 DOI: https://doi.org/10.1021/acs.jctc.8b01176

<a id ="2">[2]</a>
Grimme, S.; Bannwarth, C.; Shushkov, P. A Robust and Accurate Tight-Binding Quantum Chemical Method for Structures, Vibrational Frequencies, and Noncovalent Interactions of Large Molecular Systems Parameterized for All spd-Block Elements (Z = 1-86). J. Chem. Theory Comput. **2017**, 13 (5), 1989-2009 DOI: https://doi.org/10.1021/acs.jctc.7b00118

<a id ="3">[3]</a>
Spicher, S.; Grimme, S. Robust atomistic modeling of materials, organometallic and biochemical systems Angew. Chem. Int. Ed. **2020**, accepted article, DOI: https://doi.org/10.1002/anie.202004239

<a id ="4">[4]</a>
N. A. Teanby. 2006. An icosahedron-based method for even binning of globally distributed remote sensing data. Comput. Geosci. 32, 9 (November, 2006), 1442–1450. DOI: https://doi.org/10.1016/j.cageo.2006.01.007
