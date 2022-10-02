Supplementary material
----------------------

Supplementary material for _"Reconstructing tephra fall deposits via ensemble-based data assimilation techniques"_ by Mingari et al. (2022).

* Github repository URL: [https://github.com/lmingari/calbuco2015.git](https://github.com/lmingari/calbuco2015.git)

Datasets
--------

* Assimilation dataset: "POST_DATA/grl54177.csv"
* Validation dataset: "POST_DATA/reckziegel.csv"
* Full dataset including errors: "POST_DATA/deposit.csv"

Model configuration
-------------------

* FALL3D model input parameter file: "calbuco.inp"

Assimilation methods
--------------------

* GNC method: "POST_GNC/gnc_method.py"
* GIG method: "POST_GIG/gig_method.py"

Preprocessing scripts
---------------------

* Merge observational datasets and compute errors: "POST_DATA/clustering.py"

Postprocessing scripts
----------------------

* Plot emission profiles from the source inversion procedure: "POST_GNC/source.py"
