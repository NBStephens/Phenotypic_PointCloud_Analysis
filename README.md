# Phenotypic PointCloud Analysis
Repository for the python code in DeMars et al. 2020

If there are any questions please feel free to contact the primary authors at  nbs49@psu.edu or ldj@psu.edu

Funding information: Division of Behavioral and Cognitive Sciences, Grant/Award Numbers: 171987, 1847806; European Research Council, Grant/Award Number: 617627; RCUK BBSRC, Grant/Award Number: BB/R01292X/1 


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



