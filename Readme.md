Explore-Update Repository
===================


This repository includes code for algorithms used for PIMUS problem. There are 3 python files: one containing algorithms, and 2 others auxiliary files. 

In file **run_data.py** there are several algorithms presented. 

- Explore-Update
- Greedy
- Top-Edges
- Top-Nodes
- Brute-Force

Additionally, **run_data.py** has routines to calculate spread given a feature spread. 

In file **prepare_data.py** there are routines to:

- Convert indexes of nodes to the correct format.
- Assign probabilities on the edges according to MV and WC models.
- Write communities and node memberships to the files. 

In file **visualise_data.py** there are methods to:

- Relationship plotting (with or without double y-axis)
- Bar plotting (with or without double y-axis)

***Datasets/*** folder includes an example of dataset, models, communities, and memberships. To run algorithms with other datasets, they should be prepared in correct format, specified in these files. Given a dataset in correct format, one can obtain other necessary files (i.e. models, communities, memberships) using **prepare_data.py**. 