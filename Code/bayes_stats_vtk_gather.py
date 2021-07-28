

import os
import sys
import glob
import socket
import pathlib
import platform
import pandas as pd
import pyvista as pv
from pandas.errors import EmptyDataError
from timeit import default_timer as timer
from vtk.numpy_interface import dataset_adapter as dsa
sys.path.append(
    r'z:/RyanLab/Projects/LDoershuk/diss_pointclouds/Phenotypic_PointCloud_Analysis')

from Code.visual_utils import *
from Code.pycpd_registrations_3D import *


# Where all the single vtk files are
stats_folder = pathlib.Path(
    r"Z:\RyanLab\Projects\LDoershuk\diss_pointclouds\calcaneus\results\stats")

# The scalars you are statsing on
#scalar_list = ['BVTV', 'DA', 'Tb_Sp', 'Tb_Th', 'Tb_N']

# The output for the vtk files
vtk_out_dir = pathlib.Path(
    r"Z:\RyanLab\Projects\LDoershuk\diss_pointclouds\calcaneus\visualize")

# !!!!!!! THIS WILL NOT WORK WITHOUT SOME SORT OF LIST, HERE IT IS A ROTATION DICT THAT IS MADE BELOW> REPLACE THIS!

os.chdir(stats_folder)
current_folder_list = stats_folder.glob("*_traces")
for folders in current_folder_list:
    vtk_folders = stats_folder.joinpath(folders)
    vtk_files = [x for x in vtk_folders.rglob("*.vtk")]
    for vtk_file in vtk_files:
        shutil.copy(vtk_file, vtk_out_dir)

os.chdir(vtk_out_dir)

for vtk_file in vtk_files:
    input_mesh_name = vtk_out_dir.joinpath(f"{vtk_file}")
    output_mesh_name = f"{input_mesh_name}_consolidated"
    # Run this later in that folder consolidate_vtks() MAKE SURE YOU ARE IN THAT FOLDER... maybe
    try:
        consolidate_vtk(input_mesh=vtk_out_dir.joinpath(f"{input_mesh_name}.vtk"),
                        out_name=output_mesh_name,
                        name_match=input_mesh_name[:-4],
                        scalars=scalar_list)
    except FileNotFoundError:
        print(f"{input_mesh_name} not found!")
