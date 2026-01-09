#### Drift-Diffusion Membrane-Protein Codes

Simulation codes for "Coarse-Grained modeling of hydrodynamic behavior
in DNA synthesis monomers".

__Installation__

To install use 'pip' and 'python' version 12.1 or greater 

```
cd ./src
pip install -r requirements.txt
```

To test installation run 
```
python conc_field_01.py --help 
```

__Running the Codes__ 

Simulations can be run by using python parameter files found in the 
./cases folder for each model.  This is typically of the form
```
cd ../../src
python conc_field_01.py -p ../cases/<sim-name>/params.py
```

For more details, see the individual folders.


