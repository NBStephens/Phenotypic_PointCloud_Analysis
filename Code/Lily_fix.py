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
    r"z:/RyanLab/Projects/LDoershuk/diss_pointclouds/Phenotypic_PointCloud_Analysis"
)
from Code.pycpd_registrations_3D import *
from Code.visual_utils import *


######################################
#    Begin the actual operations     #
######################################
# To make point clouds from low res case files
# Set the input Directory
directory = pathlib.Path(r"Z:\RyanLab\Projects\LDoershuk\diss_pointclouds\talus")
os.chdir(directory)
point_distance = 2.00

# Read in the case files to a list and then sort them.
case_list = glob.glob("*aligned.vtk")
case_list.sort()

# Process them sequentially using a for loop
for files in case_list:

    # Read in the individual files and define the output name bsaed on the input name
    outName = files.replace(
        "_Trab_Out_a_femRho.case", "_BVTV_point_cloud_" + str(point_distance) + "mm.csv"
    )
    print(outName)

    # Read in the individual case file and extract the points and move the cell value to the vertices
    case = vtk_read_case(files)

    # The values are returned in a pandas dataframe, and can be written out using the to_csv funciton
    point_data = vtk_celldata_to_pointdata(case)

    # The outName is used, and we write it without the index
    point_data.to_csv(outName, index=False)


# To make high res point clouds from histomorph output vtks
# Set the input Directory
directory = pathlib.Path(
    r"Z:\RyanLab\Projects\LDoershuk\diss_pointclouds\calcaneus\hires_vtks"
)
os.chdir(directory)
point_distance = 2.00

# Read in the case files to a list and then sort them.
vtk_list = glob.glob("*aligned.vtk")
vtk_list.sort()

# Process them sequentially using a for loop
for files in vtk_list:

    # Read in the individual files and define the output name bsaed on the input name
    outName = files.replace(
        "_aligned.vtk", "_BVTV_point_cloud_" + str(point_distance) + "mm.csv"
    )
    print(outName)

    # Read in the individual case file and extract the points and move the cell value to the vertices
    vtks = pv.read(files)
    points_data = vtks.cell_data_to_point_data()
    points_array = pd.DataFrame(vtks.points)

    # The outName is used, and we write it without the index
    points_array.to_csv(outName, index=False)


# Where all the single vtk files are
nsf_folder = pathlib.Path(r"Z:\RyanLab\Projects\nsf_human_variation\Hs")

# The scalars you are statsing on
scalar_list = ["BVTV", "DA", "BSBV", "Tb_Sp", "Tb_Th", "BS", "BV", "TV", "TS", "Tb_N"]

# The output for the vtk files
vtk_out_dir = pathlib.Path(
    r"Z:\RyanLab\Projects\LDoershuk\diss_pointclouds\calcaneus\full_vtk"
)

# !!!!!!! THIS WILL NOT WORK WITHOUT SOME SORT OF LIST, HERE IT IS A ROTATION DICT THAT IS MADE BELOW> REPLACE THIS!

# This will copy all the signle vtk files to ta single folder, if your par file is fucked
for key, value in rotation_dict.items():
    current_name = key.replace("_RDN_Mesh_rotation_matrix.txt", "")
    short_name = value.replace("RDNMesh", "")
    short_name = short_name.replace("_", "")
    short_name = short_name.replace("Whole", "")
    folder_name = key.split("_")
    folder_name_parts = "/".join(folder_name[:-5])
    current_folder = nsf_folder.joinpath(folder_name_parts)
    vtk_folder = [x for x in current_folder.rglob("*.vtk")]
    vtk_folder = [x for x in vtk_folder if short_name in str(x)]
    vtk_folder = [x for x in vtk_folder if "Grid" not in str(x)]
    for vtk_file in vtk_folder:
        shutil.copy(vtk_file, vtk_out_dir)
    # short_name = value.replace("RDNRMesh", "")
    # consolidate_vtk(input_mesh = vtk_folder[0],
    # scalars=scalars, out_name=f"{current_name}_consolidated",
    #                 name_match=f"{current_name}")

os.chdir(vtk_out_dir)

for key, value in rotation_dict.items():
    input_mesh_name = value.replace("_", "")
    input_mesh_name = input_mesh_name.replace("Whole", "")
    input_mesh_name = input_mesh_name.replace("RDNMesh", "_Trab_Out_Fem_BVTV")
    output_mesh_name = f"{value}_consolidated"
    # Run this later in that folder consolidate_vtks() MAKE SURE YOU ARE IN THAT FOLDER... maybe
    try:
        consolidate_vtk(
            input_mesh=vtk_out_dir.joinpath(f"{input_mesh_name}.vtk"),
            out_name=output_mesh_name,
            name_match=input_mesh_name[:-4],
            scalars=scalar_list,
        )
    except FileNotFoundError:
        print(f"{input_mesh_name} not found!")


# To make vtk from low res case files (should not need anymore)

######################################
#    Begin the actual operations     #
######################################

# Define the directory with the point clouds and change to it.
# If on Linux/Mac use forward slashes
# directory = pathlib.Path(r"/gpfs/group/LiberalArts/default/tmr21_collab/RyanLab/Projects/nsf_human_variation/Point_cloud/Calcaneus")

# If on Windows use back slashes with an "r" in front to not that it should be read (double backslashes also work).
directory = pathlib.Path(r"Z:\RyanLab\Projects\LDoershuk\diss_pointclouds\talus")
# point_cloud_dir = pathlib.Path(r"Z:\RyanLab\Projects\LDoershuk\diss_pointclouds\calcaneus\point_clouds")

# Change to the directory with the point cloud files.
os.chdir(str(directory))

# The results for the autro3dgm matlab. Using the cortical bone alignment is the best practice
auto3d_dir = pathlib.Path(
    r"Z:\RyanLab\Projects\LDoershuk\diss_auto3dgm\talus_results\aligned"
)

# This will be inserted into each name that is generated
bone = "talus"

# Deprecated
# point_distance = "1.75"

# Define the "average\canonical" point cloud to be registered to the original point clouds.
canonical_point_cloud = glob.glob("*canonical*.csv")
print(canonical_point_cloud)

# Define the average\canonical geometry
canonical_geo = glob.glob("*canonical*.geo")
print(canonical_geo)

# Define the group names for subsetting
group_list = ["BE", "DM", "NF"]

# Define the identifying text for each of the members of the group
group1_list = [
    "BE_01",
    "BE_106",
    "BE_111",
    "BE_142",
    "BE_145",
    "BE_185",
    "BE_187",
    "BE_190",
    "BE_191",
    "BE_19A",
    "BE_20B",
    "BE_22",
    "BE_25",
    "BE_33",
    "BE_35",
    "BE_84",
    "BE_86",
    "BE_91",
    "BE_99",
]

group2_list = [
    "DM_276",
    "DM_278",
    "DM_333",
    "DM_362",
    "DM_365",
    "DM_381",
    "DM_398",
    "DM_399",
    "DM_404",
    "DM_426",
    "DM_466",
    "DM_475",
    "DM_495",
    "DM_496",
    "DM_530",
    "DM_560",
    "DM_589",
    "DM_643",
    "DM_677",
    "DM_815",
    "DM_831",
    "DM_842",
    "DM_859",
    "DM_898",
    "DM_930",
    "DM_957",
]

group3_list = [
    "NF_106",
    "NF_10",
    "NF_132",
    "NF_205",
    "NF_20",
    "NF_216",
    "NF_217",
    "NF_229",
    "NF_236",
    "NF_243",
    "NF_27",
    "NF_37",
    "NF_41",
    "NF_44",
    "NF_45",
    "NF_49",
    "NF_50",
    "NF_66",
    "NF_69",
    "NF_6",
    "NF_71",
    "NF_80",
    "NF_90",
]

# Create a list of lists for the identifiers
group_identifiers = [group1_list, group2_list, group3_list]
######################################
#             Registrations          #
######################################

# This function writes out the correct folders into the point cloud directory
setup_point_cloud_folder_struct(point_cloud_folder=directory)

# This function makes a dictionary of the expected points clouds from the rotation matrix files, and checks to make
# certain that there are as many properly named vtks in the point cloud folder
rotation_dict, vtk_list = get_initial_aligned_dict(
    auto3dgm_dir=auto3d_dir, csv_list=False
)

# This function aligns the histomorpometry vtks to the autro3dm alignment
align_vtks(
    rotation_dict=rotation_dict,
    vtk_list=vtk_list,
    point_cloud_dir=directory,
    auto3dgm_dir=auto3d_dir,
)

# This finds the point clouds that were generated from case files (which have fewer points), then vtks, and realigns them
batch_initial_rigid(
    rotation_dict=rotation_dict,
    auto3d_dir=auto3d_dir,
    point_cloud_dir=directory,
    match="short_name",
    rotation_key=True,  # This is if the rotation matrix is the key in the rotation_dict
    scalars_left=False,  # This is if you get pancake point clouds switch this to True
)
# When they are realigned, it sets the origin to 0 for all the aligned point clouds
batch_set_origin_zero(point_cloud_folder=directory)

rotation_dict_1 = {}
rotation_dict_2 = {}
for num, key_value_pairs in enumerate(rotation_dict.items()):
    print(key_value_pairs)
    if num > 34:
        rotation_dict_2[f"{key_value_pairs[0]}"] = key_value_pairs[1]
    else:
        rotation_dict_1[f"{key_value_pairs[0]}"] = key_value_pairs[1]


# This function is needed to register the low res point clouds to the high res.
# It seems they are generated slightly differently and thus need to be aligned. In the future there will be low res point
# clouds generated from the high res vtks.
# batch_register_low_and_high_res(rotation_dict=rotation_dict, point_cloud_dir=directory)
batch_register_low_and_high_res(
    rotation_dict=rotation_dict_2, point_cloud_dir=directory
)

# These next steps perform the rigid, affine, and deformable registrations automatically.
batch_rigid(
    point_cloud_dir=directory,
    canonical=canonical_point_cloud,
    iterations=100,
    tolerance=0.001,
)

batch_affine(point_cloud_dir=directory, iterations=100, tolerance=0.001)

batch_deformable(point_cloud_dir=directory, iterations=100, tolerance=0.0001)


######################################
#    Mapping and mesh generation     #
######################################

"""
Now we map the scalar values from the original to the corresponding points on the registered point cloud 
using the scipy linear interpolation. This requires a grid (i.e. the registered x, y, z) and the scalar values
from the original point cloud. Because the index has remained the same between the aligned and original point cloud
the order has remained the same. These can then be mapped taking into acount the aligned point clouds x, y, z 
to the corresponding x, y, z, of the registered point cloud. 
"""
canonical_vtk = canonical_geo[0].replace("geo", "vtk")

case_to_vtk(inputMesh=f"canonical_{bone}.case", outName=f"canonical_{bone}")

batch_mapping(
    rotation_dict=rotation_dict,
    point_cloud_dir=directory,
    canonical_vtk=canonical_vtk,
    canonical_geo=False,  # This well predates anything you are doing, leave it false
    canonical_pc=canonical_point_cloud,
)

gather_scalars(
    point_cloud_dir=directory, canonical_vtk=canonical_vtk, max_normalized=True
)

visualize_registration_movement(
    point_cloud_dir=directory, canonical_pc=canonical_point_cloud[0]
)

vtp_visualize_deformation(
    point_cloud_dir=directory,
    canonical_pc=canonical_point_cloud,
    outName=f"canonical_{bone}_analysis",
)

get_mean_vtk_groups(
    group_list=group_list,
    group_identifiers=group_identifiers,
    bone=bone,
    canonical_vtk=canonical_vtk,
    point_cloud_dir=directory,
    max_normalized=True,
)
results_dir = directory.joinpath("results")
os.chdir(results_dir)

gather_multiscalar_vtk(
    input_mesh=f"canonical_{bone}_analysis_mean.vtk",
    out_name=f"consolidated_{bone}_means.vtk",
    name_match=None,
)
