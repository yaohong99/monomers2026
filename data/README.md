#### Coarse-Grained Monomers Simulation and Analysis Codes

Simulation codes for "Coarse-Grained modeling of hydrodynamic behavior
in DNA synthesis monomers".

__Installation__

To install use 'pip' and 'python' version 12.1 or greater 

To install use LAMMPS version 2024.8

```
cd ./src
pip install -r requirements.txt
```

__Running the Codes__ 

Simulations can be run by using python parameter files found in the 
./cases folder for each model.  This is typically of the form
```
cd ../src/Boltzmann Inversion
python U.py -p ../cases/Boltzmann Inversion/B_S_values.txt
```

```
cd ../src/compare_168base
python compare_MSD.py -p ../cases/SELM/compare_168base/dirichlet_168base_X.txt
```

```
cd ../src/compare_dirichlet_2MSD
python compare_MSD.py -p ../cases/SELM/compare_dirichlet_2MSD/16base_X.txt
```

```
cd ../src/compare_period_2MSD
python compare_MSD.py -p ../cases/SELM/compare_period_2MSD/1base.txt
```

```
cd ../src/diffuison coefficient
python MSD.py -p ../cases/SELM/Diffuison Coefficient/1base.dcd
```

For more details, see the individual folders.
