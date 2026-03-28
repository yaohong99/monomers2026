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
```
cd ./data/cases/LAMMPS
```

__Running the Codes__ 

Simulations can be run by using python parameter files found in the 
./cases folder for each model.  This is typically of the form
```
cd ../src/Boltzmann Inversion
python U.py -p ../cases/Boltzmann Inversion/B_S_values.txt
```

```
CUDA_VISIBLE_DEVICES=0 nohup selm_cuda > /dev/null 2>&1 &
nohup ./lmp_mpi -in C.in  >log.txt &
```

For more details, see the individual folders.
