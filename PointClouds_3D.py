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

sys.path.append(r"z:\RyanLab\Projects\LDoershuk\diss_pointclouds\Phenotypic_PointCloud_Analysis")
from Code.pycpd_registrations_3D import *
from Code.visual_utils import *

######################################
#    Begin the actual operations     #
######################################

# Set the input Directory
directory = pathlib.Path(
    r"Z:\RyanLab\Projects\vZubritzky\Asymmetry\Medtool\point_clouds"
)
os.chdir(directory)
# point_distance = 1.75

###################################################
#    This section is only if you are              #
#    working with case files, which isn't common  #
#                                                 #
###################################################


# Read in the case files to a list and then sort them.
case_list = glob.glob("*a_femRho.case")
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


######################################
#    Begin the actual operations     #
######################################

# Define the directory with the point clouds and change to it.
# If on Linux/Mac use forward slashes
# directory = pathlib.Path(r"/gpfs/group/LiberalArts/default/tmr21_collab/RyanLab/Projects/nsf_human_variation/Point_cloud/Calcaneus")


# This will be inserted into each name that is generated
bone = "talus"

# Deprecated
# point_distance = "1.75"

# Define the "average\canonical" point cloud to be registered to the original point clouds.
canonical_point_cloud = glob.glob("*canonical*.csv")
print(canonical_point_cloud)

# Define the average\canonical geometry
# This is legacy at this point, we use vtks exclusively
canonical_geo = glob.glob("*canonical*.geo")
print(canonical_geo)

# Define the group names for subsetting
group_list = ["Lefts", "Rights"]

# Define the identifying text for each of the members of the group
group1_list = [item.name.partition("_")[0] for item in directory.glob("*L_Trab.vtk")]
group2_list = [item.name.partition("_")[0] for item in directory.glob("*R_Trab.vtk")]

# Create a list of lists for the identifiers
group_identifiers = [group1_list, group2_list]


###################################################################
#     Post analysis auto3d GM                                     #
# NOTE!!!! If you did the auto3dgm on the grey vales you need a   #
# modified workflow, which you would implement here               #
###################################################################

# The results for the autro3dgm matlab. Using the cortical bone alignment is the best practice
auto3d_dir = pathlib.Path(r"D:\Desktop\Carla\paper_3_final\rotation_matrix")

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

# This finds the point clouds that were generated from medtool, which have fewer points the vtks, and realigns them
batch_initial_rigid(
    rotation_dict=rotation_dict,
    auto3d_dir=auto3d_dir,
    point_cloud_dir=directory,
    match="short_name",
)



# When they are realigned, it sets the origin to 0 for all the aligned point clouds
batch_set_origin_zero(point_cloud_folder=directory)


# This function is needed to register the low res point clouds to the high res.
# It seems they are generated slightly differently and thus need to be aligned. In the future there will be low res point
# clouds generated from the high res vtks.
batch_register_low_and_high_res(rotation_dict=rotation_dict, point_cloud_dir=directory)


######################################
#             Registrations          #
######################################
# TODO subset full vtk point clouds to work with1
mesh = pv.read(vtk_list[0])
mesh_size = abs(np.array(mesh.bounds[::2]) - np.array(mesh.bounds[1::2]))
size = 64
x = np.linspace(mesh.bounds[0], mesh.bounds[1], size)
y = np.linspace(mesh.bounds[2], mesh.bounds[3], size)
z = np.linspace(mesh.bounds[4], mesh.bounds[5], size)
x, y, z = np.meshgrid(mesh_size)
grid = pv.StructuredGrid(x, y, z)


alg = vtk.vtkDecimatePro()
alg.SetInputData(poly_data)
alg.SetTargetReduction(reduction)
alg.SetPreserveTopology(preserve_topology)
alg.SetFeatureAngle(feature_angle)
alg.SetSplitting(splitting)
alg.SetSplitAngle(split_angle)
alg.SetPreSplitMesh(pre_split_mesh)
alg.Update()

mesh = _get_output(alg)
if inplace:
    poly_data.overwrite(mesh)
else:
    return mesh


result = grid.interpolate(mesh, radius=20.0)

new_mesh = pv.create_grid(
    mesh, dimensions=(mesh_size[0] / 2, mesh_size / 2, mesh_size / 2)
)
result = mesh.sample(new_mesh)

pro_decimated = mesh.decimate_pro(target_reduction, preserve_topology=True)


rotation_dict, vtk_list = get_rotation_dict(histomorph_vtk_dir=directory)

# This function writes out the correct folders into the point cloud directory
setup_point_cloud_folder_struct(point_cloud_folder=directory)


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

case_to_vtk(inputMesh="canonical_radius.case", outName="canonical_radius")

batch_mapping(
    rotation_dict=rotation_dict,
    point_cloud_dir=directory,
    canonical_vtk=canonical_vtk,
    canonical_geo=False,
    canonical_pc=canonical_point_cloud,
)

gather_scalars(
    point_cloud_dir=directory, canonical_vtk=canonical_vtk, max_normalized=False
)

visualize_registration_movement(
    point_cloud_dir=directory, canonical_pc=canonical_point_cloud[0]
)

vtp_visualize_deformation(
    point_cloud_dir=directory,
    canonical_pc=canonical_point_cloud,
    outName="canonical_talus_analysis_3",
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
    input_mesh="canonical_talus_analysis_3_mean.vtk",
    out_name="consolidated_means.vtk",
    name_match=None,
)
