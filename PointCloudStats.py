import os
import sys
import glob
import socket
import pathlib
import platform
import pandas as pd
import pyvista as pv

# script_dir = pathlib.Path(r"C:\Users\skk5802\Desktop\Phenotypic_PointCloud_Analysis")
sys.path.append(r"D:\git_pulls\Phenotypic_PointCloud_Analysis")
# script_dir = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
# sys.path.append(str(script_dir))
from Code.pycpd_registrations_3D import *
from Code.get_pvalues_point_cloud import *
from Code.PPCA_utils import _end_timer, _get_outDir, _vtk_print_mesh_info


############################################################
#                                                          #
#                   Being operations                       #
#                                                          #
############################################################
# Define the directory where the files are
directory = pathlib.Path(r"D:\Desktop\Carla\paper_3_final")
os.chdir(directory)

bone = "talus"

# Define the group names for subsetting
group_list = ["D", "E", "F"]

# Define the identifying text for each of the members of the group
group1_list = [
    "BeliManastir_G20",
    "BO_1_M",
    "BO_11_F",
    "BO_4_F",
    "BO_5_F",
    "BO_6_F",
    "Ilok_G72",
    "Ilok_G70",
    "NF_821012",
    "Paks_1164",
    "Paks_1166",
    "Paks_1846",
    "Paks_997",
    "PARMA_7_F",
    "PerkataNyuli_4263",
    "Velia_T209",
    "Velia_T342",
    "Velia_T375",
    "Velia_T390",
]
group2_list = [
    "BeliManastir_G31",
    "BO_40_M",
    "BO_6_M",
    "Paks_1865",
    "PerkataNyuli_1575",
    "PerkataNyuli_2123",
    "PerkataNyuli_435",
    "PerkataNyuli_752",
    "Velia_T138",
    "Velia_T320",
    "Velia_T333",
]
group3_list = [
    "BeliManastir_G4",
    "BO_39_M",
    "Paks_1156",
    "PARMA_10_F",
    "PerkataNyuli_116",
    "PerkataNyuli_734",
    "BO_81_M",
    "Velia_T365",
]

# Create a list of lists for the identifiers
group_identifiers = [group1_list, group2_list, group3_list]
################################################
# Shouldn't need to modify anything below here #
################################################

outName = "canonical_" + str(bone)
print(f"Prefix for file names will be {outName}")

point_cloud_model, canonical_geo, canonical_ply = get_necessary_files(
    matching_text="canonical"
)

# Read it in so we can get the resel value and uvalue for the pvalue thresholding
point_cloud_model = pd.read_csv(point_cloud_model, header=None)

case_to_vtk(inputMesh=f"{canonical_geo[:-4]}.case", outName=outName)
################################################
# For cortical bone or 2d surface mesh         #
################################################

resels = get_resels_2d(point_cloud_model=point_cloud_model, mesh=canonical_ply)
print(f"Resels estimated at {resels}...")

get_data_files_for_scalars(
    group_list=group_list,
    group_identifiers=group_identifiers,
    point_cloud_dir=pathlib.Path.cwd(),
    scalar_list=["CtTh"],
    max_normalixed=True,
)

get_ttest_results_for_scalars(
    point_cloud_dir=pathlib.Path.cwd(),
    scalar_list=["CtTh"],
    canonical_geo=canonical_geo,
    resels=resels,
    pvalue=0.05,
    max_normalized=True,
)

consolidate_case(
    inputMesh=f"CtTh_max_normalized_ttest_results_Anteaters_vs_KWapes_Tscore.case",
    outName=f"{bone}_consolidated_CtTh_ttest",
    nameMatch=group_list,
    scalars=["CtTh"],
    outputs=["_Tscore"],
    max_normazlied=True,
    pairwise=True,
)

################################################
# For trabecular bone or other 3d volume mesh  #
################################################

# Get an estimation of the resolutions elements from the mean point cloud
resels = get_resels_3d(point_cloud_model=point_cloud_model, mesh=canonical_ply)
print(f"Resels estimated at {resels}...")

get_data_files_for_scalars(
    group_list=group_list,
    group_identifiers=group_identifiers,
    point_cloud_dir=pathlib.Path.cwd(),
    scalar_list=["BVTV", "DA"],
    max_normalixed=True,
)

get_ttest_results_for_scalars(
    point_cloud_dir=pathlib.Path.cwd(),
    scalar_list=["BVTV", "DA"],
    canonical_geo=canonical_geo,
    resels=resels,
    pvalue=0.05,
    max_normalized=True,
)

#### If you are working from VTK's this is the proper method.
canonical_vtk = glob.glob("canonical*.vtk")[0]

get_data_files_for_vtk_scalars(
    group_list=group_list,
    group_identifiers=group_identifiers,
    point_cloud_dir=directory,
    max_normalixed=True,
)

results_vtk = ttest_comparison_vtk(
    canonical_vtk=canonical_vtk, resels=resels, point_cloud_dir=directory, pvalue=0.05
)
results_vtk.save("talus_t_tests_analysis_3.vtk")
