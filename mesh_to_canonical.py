import os
import sys
import glob
import socket
import pathlib
import platform

if platform.system() == "Windows":
    if socket.gethostname() == 'L2ANTH-WT0023':
        sys.path.append(r"Z:\RyanLab\Projects\NStephens\git_repo")
    else:
        sys.path.append(r"D:\Desktop\git_repo")
if platform.system().lower() == 'linux':
    if 'redhat' in platform.platform():
        sys.path.append(r"/gpfs/group/LiberalArts/default/tmr21_collab/RyanLab/Projects/NStephens/git_repo")
    else:
        sys.path.append(r"/mnt/ics/RyanLab/Projects/NStephens/git_repo")

from MARS.morphology.Mesh_2d_to_3d import *

###################################
#                                 #
# This is for where you do stuff  #
#                                 #
###################################

# Set up a temporary local directory, which is necessary when working on a cluster where you may not have permissions
temp = pathlib.Path(tempfile.gettempdir())

# Define the directory where the ply/off file is
directory = pathlib.Path(r"D:\Desktop\For_Jaap")

# This is where tetwild is located, which is platform specific (i.e. Linux or Windows or cluster)
tetwildpath = pathlib.Path(r"D:\Desktop\git_repo\MARS\TetWild")

# Change to the directory
os.chdir(directory)

#Define the bone, which will become part of the name
bone = "Talus"

##########################################################################
#                                                                        #
# This is for 2D canonicals from a triangular mesh (i.e. cortical bone)  #
#                                                                        #
##########################################################################

input_name = glob.glob("*Trabecular_meshROI*.ply")

ply_to_inp(inMesh=input_name[0], outName=f"canonical_{bone}", outDir=directory)

# Define an inp input. If you have more than one inp, it will find them all, but only select the first.
# Be more specific if you need to work with a certain file.
inp_file = glob.glob(f"canonical_{bone}.inp")[0]
print(inp_file)

# This is for cortical bone and will write out a surface case file
inp_to_case_2d(in_name=inp_file, outname=f"canonical_{bone}")

# The key at this point is to make certain you don't have too many vertices, since this will make the calculation
# time in pycpd overly long. To avoid this, it is recommended that you use meshlab to simplify and smooth the mesh prior.

###################################################################################################################
#                                                                                                                 #
#      This is for 3D canonicals, where a triangular mesh is converted (voxelized) into a tetrahedral mesh        #
#      (i.e. trabecular bone)                                                                                     #
#                                                                                                                 #
###################################################################################################################


#
#  !!!! Make sure you saved the mesh in meshlab. It seems to hate meshes saved in cloud compare !!!!    #
#


# Define the mesh file to import
input_name = glob.glob("*canonical*.ply")
input_name.sort()
input_name = input_name[0]
print(input_name)

#This uses shutil to copy the input ply and place it in the local temp folder so tetwild can voxelize it.
dest = shutil.copy(input_name, temp)

# Use tetwild to voxelize the 2d mesh. If you are doing this in Spyder you won't see much going on, because it outputs
# to the terminal.
TetWild_2d_to_3d(input_path=str(pathlib.Path(temp)),
                 in_file=input_name,
                 output_path=str(pathlib.Path(temp)),
                 tetwildpath=str(tetwildpath),
                 out_name="",
                 edge_length="",
                 #target_verts=3000, #You can set the target number of vertices, but it doesn't adjust the surface cells.
                 laplacian=True)



#The -3 removes the last three characters in a string, which allows us to place an inp at the end.
out_file = str(input_name[:-3]) + "inp"
out_file = pathlib.Path(temp).joinpath(out_file)

# Copy the voxelized mesh to the original directory
dest = shutil.copy(out_file, directory)

inp_to_case(out_file, f"canonical_{bone}")