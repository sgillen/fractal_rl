# Shrinking Fractional Dimensions With Reinforcement Learning

This directory contains the code required to replicate the results presented in the manuscript.

- common.py - Contains functions needed everywhere, this includes algorithms 1 and 2 from the paper.
- ars.py - Our implementation of ars.
- run_vardim.py - Code to run ars with the variation dimension post processors, as presented in figure 1 of the paper.
- run_mdim.py - Code to run ars with the mesh dimension post proceessors, as presented in figure 2 of the paper.
- make_plots_and_tables.ipynb - Jupyter notebook used to create the data for the tables presented in the paper.
- requirments.txt - Requirements file to install dependencies.

- data17/* results from the variation dimension experiments presented in the paper.
- data_mcshdim4/ results from the mesh dimension experiments presented in the paper.


Unfortunatley, replicating the results requires a mujoco license, we use version 1.51 here. If you have that available you can do the following:

First create a new python environment, in conda this would be:
```
conda create --name corl_repl python=3.6
conda activate corl_repl
```

Then install the requirements in the new environment:
```
pip install -r requirements.txt 
```

To run the jupyter notebook, install the new environment with:
```
python -m ipykernel install --user --name corl_repl
```

From here you can run all the files provided. run_vardim.py should generate a copy of data17/ if you have enough patience, and run_mdim.py should generate data_mcshdim4/.
The notebook will load this data and provide pltos and the data presented in the paper.
