# Phenotypic PointCloud Analysis
Repository for the python code in DeMars et al. 2020



<a href="https://zenodo.org/badge/latestdoi/290870082"><img src="https://zenodo.org/badge/290870082.svg" alt="DOI"></a>

If there are any questions please feel free to contact the primary authors at nbs49@psu.edu or lvd5263@psu.edu

Funding information: Division of Behavioral and Cognitive Sciences, Grant/Award Numbers: 171987, 1847806; European Research Council, Grant/Award Number: 617627; RCUK BBSRC, Grant/Award Number: BB/R01292X/1 

### Description

The Phenotypic PointCloud Analysis is an open source python worflow that allows for the three dimensional registration, statistical analysis, and visualization of scalar data associated with points generated from tetrahedral meshes. This worfklow utilizes the coherent point drift algorithm or Myronenko and Song, 2010 ( 10.1109/tpami.2010.46) to perform a rigid, affine, and deformable registration of a mean shape to inviddual points clouds. Thereafter the scalar information associated with the individual point clouds is statistically analyzed following the statistical parametric methods describved by Worsley et al. 1996 (10.1002/(sici)1097-0193(1996)4:1<58::aid-hbm4>3.0.co;2-o) and Friston et al. 1994 (10.1002/hbm.460020402). The statistical results may then be visualized in 3d on a tetrahedral mesh of the mean morphology in various freely available visualization platforms (e.g. paraview). This method was most recently demonstrated in DeMars et al 2020 (10.1002/ajhb.23468), using trabecular and cortical bone values derived from micro-CT scans of human calcanei. If you use this code in any of your work, please cite the described sources and the following paper:


```
@Article{DeMars2020,
  author    = {Lily J. D. DeMars and Nicholas B. Stephens and Jaap P. P. Saers and Adam Gordon and Jay T. Stock and Timothy M. Ryan},
  journal   = {American Journal of Human Biology},
  title     = {Using point clouds to investigate the relationship between trabecular bone phenotype and behavior: An example utilizing the human calcaneus},
  year      = {2020},
  month     = {aug},
  pages     = {e23468},
  doi       = {10.1002/ajhb.23468},
  publisher = {Wiley},
}
```
Lily J. D. DeMars


Nicholas B. Stephens <div itemscope itemtype="https://schema.org/Person"><a itemprop="sameAs" content="https://orcid.org/0000-0002-2838-5606" href="https://orcid.org/0000-0002-2838-5606" target="orcid.widget" rel="me noopener noreferrer" style="vertical-align:top;"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon">https://orcid.org/0000-0002-2838-5606</a></div>

### Installing

Recommended Software: 
Anaconda (Python distribution with many useful packages installed) 
A fully featured text editor or IDE (Mostly for checking syntax when doing local adjustments to scripts.)
Paraview (Free visualization and analystical program for various file types)
CloudCompare (Free program for point cloud/mesh processing)

The easiest method for install is to create a conda environment from the yaml file included in the setup directory. 
This also ensures that package versions used in a local environment are the same that these scripts were devleoped with. 

If you are on Windows, you simply have to double click the install_PPCA.bat file.

If you are on Linux/MAC you can install by runng the install_PPCA.sh file. 

Note that you may have to give the file permissions to execute as a program (right click) or through the command line:

```
cd /where/the/file/is/
chmod 755 install_MARS.sh
./install_MARS.sh
```

Alternatively you can use conda to install the environment from the command line in the following manner:

```
conda env create --name=PPCA --file=/where/the/yaml/file/is/PPCA_conda.yaml
```

### Useage

The virtual environment can then be accessed through the command line in a jupyter console/qt-console

```
conda activate PPCA
jupyter-qtconsole

```

However, if you are coming from an RStudio background, the Spyder IDE may be a better choice:

```
conda activate PPCA
spyder
```

Base scripts for various work flows are maintained in the ./Scripts folder (PointClouds_3D) but there are many funcitons that may be adapted to your work.


You can see examples of the workflow here:



Publications using these routines: 



