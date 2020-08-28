import os
import sys
import glob
import socket
import pathlib
import platform
import pandas as pd
import pyvista as pv
from Phenotypic_PointCloud_Analysis.code.pycpd_registrations_3D import *
from Phenotypic_PointCloud_Analysis.code.get_pvalues_point_cloud import *

############################################################
#                                                          #
#                   Being operations                       #
#                                                          #
############################################################
#Define the directory where the files are
directory = pathlib.Path(r"D:\Desktop\Sharon\New_Point_Cloud_cortical")
os.chdir(directory)

bone = "radius"

# Define the group names for subsetting
group_list = ["KWapes", "Anteaters", "OWM", "nonKWapes"]

# Define the identifying text for each of the members of the group
group1_list = ["51202", "51393", "51377", "201588", "51379", "54091"]
group2_list = ["23437", "262655", "61795", "211662", "23436", "100068", "133490"]
group3_list = ["82096", "Papio_ursinusPapioursinus", "80774", "82097", "43086", "28256", "34712", "52206", "52223", "34714", "169430", "89365"]
group4_list = ["NF821349", "NF821350", "NF821282", "NF821211", "200898", "NF819955", "NF819953", "NF819951", "NF820715"]

# Create a list of lists for the identifiers
group_identifiers = [group1_list, group2_list, group3_list, group4_list]

################################################
# Shouldn't need to modify anything below here #
################################################

outName = "canonical_" + str(bone)
print(f"Prefix for file names will be {outName}")

point_cloud_model, canonical_geo, canonical_ply = get_necessary_files(matching_text="canonical")

#Read it in so we can get the resel value and uvalue for the pvalue thresholding
point_cloud_model = pd.read_csv(point_cloud_model, header=None)


################################################
# For cortical bone or 2d surface mesh         #
################################################

resels = get_resels_2d(point_cloud_model=point_cloud_model, mesh=canonical_ply)
print(f"Resels estimated at {resels}...")

get_data_files_for_scalars(group_list=group_list,
                           group_identifiers=group_identifiers,
                           point_cloud_dir=pathlib.Path.cwd(),
                           scalar_list=["CtTh"],
                           max_normalixed=True)

get_ttest_results_for_scalars(point_cloud_dir=pathlib.Path.cwd(),
                              scalar_list=["CtTh"],
                              canonical_geo=canonical_geo,
                              resels=resels,
                              pvalue=0.05,
                              max_normalized=True)


################################################
# For trabecular bone or other 3d volume mesh  #
################################################

#Get an estimation of the resolutions elements from the mean point cloud
resels = get_resels_3d(point_cloud_model=point_cloud_model, mesh=canonical_ply)
print(f"Resels estimated at {resels}...")

get_data_files_for_scalars(group_list=group_list,
                           group_identifiers=group_identifiers,
                           point_cloud_dir=pathlib.Path.cwd(),
                           scalar_list=["BVTV", "DA"],
                           max_normalixed=True)

get_ttest_results_for_scalars(point_cloud_dir=pathlib.Path.cwd(),
                              scalar_list=["BVTV", "DA"],
                              canonical_geo=canonical_geo,
                              resels=resels,
                              pvalue=0.05,
                              max_normalized=True)

#### If you are working from VTK's this is the proper method.
canonical_vtk = glob.glob("canonical*.vtk")[0]

get_data_files_for_vtk_scalars(group_list=group_list, group_identifiers=group_identifiers,
                               point_cloud_dir=directory, max_normalixed=True)


ttest_comparison_vtk(canonical_vtk=canonical_vtk, resels=resels, point_cloud_dir=directory, pvalue=0.05)