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

if platform.system() == "Windows":
    if socket.gethostname() == "Irreverent":
        sys.path.append(r"D:\Desktop\git_repo")
    else:
        sys.path.append(r"Z:\RyanLab\Projects\NStephens\git_repo")

if platform.system() == 'Linux':
    if 'redhat' in platform.platform():
        sys.path.append(r"/gpfs/group/LiberalArts/default/tmr21_collab/RyanLab/Projects/NStephens/git_repo")
else:
        sys.path.append(r"/mnt/ics/RyanLab/Projects/NStephens/git_repo")
from MARS.morphology.vtk_mesh import *
from MARS.registration.pycpd_registrations_3D import *
from MARS.utils.MARS_utils import _end_timer

######################################
#    Begin the actual operations     #
######################################

#Set the input Directory
directory = pathlib.Path(r"D:\Desktop\Sharon\New_Point_Cloud_trabecular")
os.chdir(directory)
point_distance = 1.75

#Read in the case files to a list and then sort them.
case_list = glob.glob("*a_femRho.case")
case_list.sort()

#Process them sequentially using a for loop
for files in case_list:

    #Read in the individual files and define the output name bsaed on the input name
    outName = files.replace("_Trab_Out_a_femRho.case", "_BVTV_point_cloud_" + str(point_distance) + "mm.csv")
    print(outName)

    #Read in the individual case file and extract the points and move the cell value to the vertices
    case = vtk_read_case(files)

    #The values are returned in a pandas dataframe, and can be written out using the to_csv funciton
    point_data = vtk_celldata_to_pointdata(case)

    #The outName is used, and we write it without the index
    point_data.to_csv(outName, index=False)


######################################
#    Begin the actual operations     #
######################################

#Define the directory with the point clouds and change to it.
#If on Linux/Mac use forward slashes
#directory = pathlib.Path(r"/gpfs/group/LiberalArts/default/tmr21_collab/RyanLab/Projects/nsf_human_variation/Point_cloud/Calcaneus")

#If on Windows use back slashes with an "r" in front to not that it should be read (double backslashes also work).
directory = pathlib.Path(r"D:\Desktop\Sharon\New_Point_Cloud_trabecular")

#Change to the directory with the point cloud files.
os.chdir(str(directory))

#The results for the autro3dgm matlab. Using the cortical bone alignment is the best practice
auto3d_dir = pathlib.Path(r"D:\Desktop\Sharon\Cortical_results\aligned")

#This will be inserted into each name that is generated
bone = "radius"

#Deprecated
#point_distance = "1.75"

#Define the "average\canonical" point cloud to be registered to the original point clouds.
canonical_point_cloud = glob.glob("*canonical*.csv")
print(canonical_point_cloud)

#Define the average\canonical geometry
canonical_geo = glob.glob("*canonical*.geo")
print(canonical_geo)

# Define the group names for subsetting
group_list = ["KWapes", "Anteaters", "OWM", "nonKWapes"]

# Define the identifying text for each of the members of the group
group1_list = ["51202", "51393", "51377", "201588", "51379", "54091"]
group2_list = ["23437", "262655", "61795", "211662", "23436", "100068", "133490"]
group3_list = ["82096", "Papio_ursinusPapioursinus", "80774", "82097", "43086", "28256", "34712", "52206", "52223", "34714", "169430", "89365"]
group4_list = ["NF821349", "NF821350", "NF821282", "NF821211", "200898", "NF819955", "NF819953", "NF819951", "NF820715"]

# Create a list of lists for the identifiers
group_identifiers = [group1_list, group2_list, group3_list, group4_list]

######################################
#             Registrations          #
######################################

# This function writes out the correct folders into the point cloud directory
setup_point_cloud_folder_struct(point_cloud_folder=directory)

# This function makes a dictionary of the expected points clouds from the rotation matrix files, and checks to make
# certain that there are as many properly named vtks in the point cloud folder
rotation_dict, vtk_list = get_initial_aligned_dict(auto3dgm_dir=auto3d_dir)

# This function aligns the histomorpometry vtks to the autro3dm alignment
align_vtks(rotation_dict=rotation_dict, vtk_list=vtk_list, point_cloud_dir=directory, auto3dgm_dir=auto3d_dir)

# This finds the point clouds that were generated from medtool, which have fewer points the vtks, and realigns them
batch_initial_rigid(rotation_dict=rotation_dict, auto3d_dir=auto3d_dir, point_cloud_dir=directory, match="long_name")

# When they are realigned, it sets the origin to 0 for all the aligned point clouds
batch_set_origin_zero(point_cloud_folder=directory)

# This function is needed to register the low res point clouds to the high res.
# It seems they are generated slightly differently and thus need to be aligned. In the future there will be low res point
# clouds generated from the high res vtks.
batch_register_low_and_high_res(rotation_dict=rotation_dict, point_cloud_dir=directory)

# These next steps perform the rigid, affine, and deformable registrations automatically.
batch_rigid(point_cloud_dir=directory, canonical=canonical_point_cloud, iterations=100, tolerance=0.001)

batch_affine(point_cloud_dir=directory, iterations=100, tolerance=0.001)

batch_deformable(point_cloud_dir=directory, iterations=100, tolerance=0.001)


######################################
#    Mapping and mesh generation     #
######################################

'''
Now we map the scalar values from the original to the corresponding points on the registered point cloud 
using the scipy linear interpolation. This requires a grid (i.e. the registered x, y, z) and the scalar values
from the original point cloud. Because the index has remained the same between the aligned and original point cloud
the order has remained the same. These can then be mapped taking into acount the aligned point clouds x, y, z 
to the corresponding x, y, z, of the registered point cloud. 
'''
canonical_vtk = canonical_geo[0].replace("geo", "vtk")

case_to_vtk(inputMesh="canonical_radius.case", outName="canonical_radius")

batch_mapping(rotation_dict=rotation_dict, point_cloud_dir=directory,
              canonical_geo=canonical_geo, canonical_pc=canonical_point_cloud)

gather_scalars(point_cloud_dir=directory, canonical_vtk=canonical_vtk, max_normalized=True)

visualize_registration_movement(point_cloud_dir=directory, canonical_pc=canonical_point_cloud[0])

vtp_visualize_deformation(point_cloud_dir=directory, canonical_pc=canonical_point_cloud, outName="canonical_radius")

get_mean_vtk_groups(group_list=group_list, group_identifiers=group_identifiers, bone=bone, canonical_vtk=canonical_vtk,
                    point_cloud_dir=directory, max_normalized=True)
