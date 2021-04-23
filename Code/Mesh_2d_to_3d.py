'''
Script to take a 2d ply/off and convert it into a 3d volume mesh. It is recommened that the mesh be viewed and fixed
(e.g. remove floating vertices, creases, etc.) in meshlab prior to conversion.

If you have not already, you will need to install the following packages:
conda install -c conda-forge python-gmsh
pip install gmsh-sdk

Author: Nicholas Stephens (nbs49@psu.edu)
Date: 06/21/2019

'''

import os
import sys
import gmsh
import glob
import shutil
import meshio
import trimesh
import pathlib
import tempfile
import platform
import subprocess
import numpy as np
import pandas as pd
import pyvista as pv
from time import sleep
from timeit import default_timer as timer

def mesh_info(mesh):
    """
    A function to return basic information about a loaded 2d mesh
    :param mesh: A 2d mesh loaded through trimesh.load
    """
    mesh_2d = mesh
    triangles = len(mesh_2d.triangles)
    points = len(mesh_2d.vertices)
    edgemin = mesh_2d.edges_unique_length.min()
    edgemax = mesh_2d.edges_unique_length.max()
    edgemean = mesh_2d.edges_unique_length.mean()
    if mesh_2d.is_volume == True:
        print("Mesh can be represented as a 3d volume.")
    else:
        if mesh_2d.is_watertight != True:
            print("Mesh has holes.")
        else:
            print("Mesh doesn't have holes, please check for other issues in Meshlab.")
    print("Mesh has {} faces and {} vertices.".format(triangles, points))
    print("Mesh edge length: \n           mean {:06.4f}, max {:06.4f}, min {:06.4f}".format(edgemean, edgemax, edgemin))

#Function to read in an inp and output a case
def inp_to_case(in_name, outname):
    """
    Function to read in an ascii inp file and output a point cloud, for cloud compare, and a case file,
    readable by paraview.

    :param in_name: Abaqus Ascii inp file name.
    :param outname: Output file name
    :return:
    """
    start = timer()
    # Open the inp and find the diagnostic lines
    with open(in_name, "rt") as f:
        # Get the number of lines in the file
        size = len([0 for _ in f])
        print("Lines in inp file:", size)
    with open(in_name, "rt") as f:
        # Get the number of lines in the file
        #start_xyz = "*NODE"
        start_xyz = "*Node"
        #start_elements = "******* E L E M E N T S *************"
        start_elements = "*Element"
        start_element_sets = "*ELSET"
        for num, line in enumerate(f, 1):
            if start_xyz in line:
                print('Node set starts at line:', num)
                print(line)
                xyz = num
            if start_elements in line:
                print('Elements start at line:', num)
                print(line)
                elements = num
            if start_element_sets in line:
                print('Element sets start at line:', num)
                print(line)
                sets = num
        f.close()

    # Use the line values to define the skip rows list for pandas

    # Start of file to nodes line
    range_1 = list(range(0, int(xyz)))

    # Start of element line to end of file.
    range_2 = list(range(int(elements - 1), int(size)))

    # Start of file until the end of the elements.
    range_3 = list(range(int(0), int(elements)))

    # Start of the element set until the end of the file.
    try:
        range_4 = list(range(int(sets - 1), int(size)))
        set_range = list(range(int(0), int(sets)))
    except:
        range_4 = list(range(int(size) - 1, int(size)))
        print("No element sets found.")

    # Create a list with the individual range lists
    xyz_range = range_1 + range_2

    element_range = range_3 + range_4

    # Read in individual dataframes for the portions. It isn't efficient, but it is readable.
    xyz_df = pd.read_csv(in_name, header=None, sep=",", skiprows=xyz_range)

    # Use ravel to stack the rows untop of one another
    stacked_xyz = pd.DataFrame(xyz_df.iloc[:, 1:4].values.ravel('F'))

    # Define the elements dataframe
    elements_df = pd.read_csv(in_name, header=None, sep=",", index_col=False, skiprows=element_range)
    elements_df = elements_df.astype(int)

    with open(outname + ".case", 'w', newline="") as fout:
        # tells where to find the geometry (i.e. .geo file) and what format
        fout.write('FORMAT\ntype: ensight gold\n\nGEOMETRY\nmodel:                           ')
        # Scalars per node for 3d geometry
        fout.write(outname + ".geo\n")
        fout.close()
    with open(outname + '.geo', 'w', newline="") as fout:
        # Case header defining the name of the scalar array
        Title = str(outname) + ".inp"
        # fout to write the header
        fout.write("Title:  " + str(Title) + "\n")
        # fout to write the shape of the tetrahedral geometry (i.e. tetra4)
        fout.write('Description 2\nnode id given\nelement id given\npart\n         1\n')
        fout.write('Description PART\ncoordinates \n     ' + str(len(xyz_df)) + "\n")
        # use pandas in conjunction with fout to append the scalars
        xyz_df[0].to_csv(fout, header=False, index=False, sep=" ")
        # Write out the case format header with fout
        stacked_xyz.to_csv(fout, header=False, index=False, sep=" ")
        # Write out the elments
        fout.write("\ntetra4\n" + str(len(elements_df)) + "\n")
        elements_df[0].to_csv(fout, header=False, index=False, sep=" ")
        elements_df.iloc[:, 1:].to_csv(fout, header=False, index=False, sep=" ")
        fout.close()

    #Write out the point cloud
    xyz_df.iloc[:, 1:4].to_csv(outname + "_pointcloud.csv", header=None, index=False)

    end = timer()
    end_time = (end - start)
    print("Coversion took", end_time, "\n")

def TetWild_2d_to_3d(input_path, in_file, output_path, tetwildpath, out_name="", edge_length="", target_verts="", laplacian=False):
    """
    Tetrahedralize from a 2d .off, .obj, .stl, or .ply file. Output is in .msh/.mesh format. The .msh is then converted to an inp using gmsh.
    It is assumed that gmsh is installed and on the system path.
    See Hu Yixin, et al. 2018 (http://doi.acm.org/10.1145/3197517.3201353) and https://github.com/Yixin-Hu/TetWild for further information.
    :param input_path:
    :param in_file:
    :param output_path:
    :param tetwildpath:
    :param out_name:
    :param edge_length:
    :param target_verts:
    :param laplacian:
    :return:
    """
    #Reads in the input file name as a string.
    in_file = str(in_file)

    if out_name == "":
        out_file = str(in_file[:-3]) + "inp"
    else:
        out_file = str(out_name) + ".inp"
        out_file = out_file.replace("..", ".")

    #Removes the last 3 characters and appends the new output type
    msh_file = str(in_file[:-4]) + "_.msh"

    #Gives the long file path and name for reading in
    in_file = pathlib.Path(input_path).joinpath(in_file)

    #in_file = pathlib.Path(input_path).joinpath(in_file)
    print("Input:", in_file)

    #output = pathlib.Path(input_path).joinpath("lowres")
    output = pathlib.Path(output_path).joinpath(out_file)
    print("Output: ", output)

    #This is where meshlabserver lives
    tetwildpath = pathlib.Path(tetwildpath).joinpath("TetWild.exe")

    #So this is just putting it into a format that meshlab wants. -i is input, -o is output -s is the script to use
    command = '"' + str(tetwildpath) + '"' + " " + str(in_file)
    if edge_length != "":
        edge_length = float(edge_length)
        command += " --ideal-edge-length " + str(edge_length)
    if target_verts != "":
        target_verts = int(target_verts)
        command += " --targeted-num-v " + str(target_verts)
    if laplacian == True:
        command += " --is-laplacian "

    #Tells us the command we are sending to the console
    print("\nGoing to execute:\n                            {}".format(str(command)))

    #Actually sends this to the console
    output = subprocess.call(command)

    #This SHOULD show us the error messages or whatever....
    last_line = output
    print(last_line)
    print("\n\nDone!\n")

    temp = pathlib.Path(tempfile.gettempdir())
    msh_file = pathlib.Path(temp).joinpath(msh_file)
    out_file = pathlib.Path(temp).joinpath(out_file)


    print("\nConverting msh to inp:\n                      {}".format(str(out_file)))
    print("\n")
    sleep(.5)
    get_gmsh_info(msh_file)
    print("\n")

    # Use meshio to read in the msh file and output the inp. Easier than hoping gmsh installed properly.
    mesh = meshio.read(str(msh_file))
    mesh.write(str(out_file))

def ply_to_inp(inMesh, outName="", outDir=""):
    """
    Function to convert a ply file to an inp file using pyvista and meshio.
    :param inMesh: Input ply file name (ascii or binary)
    :param outName: The desired output name. If an empty string ("") is given, the input file name will be used.
    :param outDir: Where the inp should be saved to. If an empty string ("") is given, the current directory will be used.
    :return: Returns an inp mesh file.
    """

    # Make sure we are dealing with a string
    inMesh = str(inMesh)

    #If the blank string is passed, then remove the last 4 characters
    if outName == "":
        outName = inMesh[:-4]

    # If the user happens to place a .inp on the end we can just replace it so it doesn't end up being .inp.inp
    outName = outName.replace(".inp", "")

    #If there is no outDir then you can use pathlib to get the current working directory
    if outDir == "":
        outDir = pathlib.Path.cwd()

    # Use pathlib joinpath. This keeps track of which direction your slashes are going
    outName = outDir.joinpath(outName)

    outDir = str(outDir)

    # Read in the mesh with pyvista. Meshio itself seems to hate ply files.
    # Pyvista is fine reading the ply, and it handles the internal conversion to something meshio likes.
    mesh = pv.read(filename=inMesh, file_format="ply")
    print(mesh)
    print(f"Saving {inMesh[:-4]} as an inp file to {str(outDir)}")
    pv.save_meshio(f"{outName}.inp", mesh)

def _find_string(inFile, search_string, message=""):
    if type(search_string) == list:
        pass
    if message == "":
        message = search_string
    with open(inFile, "rt") as file:
        for number, line in enumerate(file):
            if str(search_string) in line:
                print(f'{message} starts at line {number}')
                print(line)
                line_number = number
        return line_number

#Function to read in an inp and output a case
def inp_to_case_2d(in_name, outname):
    '''
    Function to read in an ascii inp file and output a point cloud, for cloud compare, and a case file,
    readable by paraview.

    :param in_name: Abaqus Ascii inp file name.
    :param outname: Output file name
    :return:
    '''

    start = timer()
    # Open the inp and find the diagnostic lines
    with open(in_name, "rt") as file:
        # Get the number of lines in the file
        size = len([0 for _ in file])
        print(f"There are {size} lines in inp file.")
        # Get the line number for each text match in the file
    file.close()

    # This takes a while because it is opening and reopening files. A cleaner solution is needed
    try:
        xyz = _find_string(inFile=in_name, search_string="*NODE", message="")
        range_1 = list(range(0, int(xyz) + 1))
    except:
        xyz = _find_string(inFile=in_name, search_string="*Node", message="")
        range_1 = list(range(0, int(xyz) + 1))
    try:
        elements = _find_string(inFile=in_name, search_string="******* E L E M E N T S *************", message="")
    except:
        elements = _find_string(inFile=in_name, search_string="*Element", message="")
    try:
        sets = _find_string(inFile=in_name, search_string="*ELSET", message="")
    except:
        sets = size

    range_2 = list(range(int(elements), int(size + 1)))

    # Start of file until the end of the elements.
    range_3 = list(range(int(0), int(elements + 1)))

    # Start of the element set until the end of the file.
    range_4 = list(range(int(sets - 1), int(size)))

    # Create a list with the individual range lists
    xyz_range = range_1 + range_2
    element_range = range_3 + range_4

    # Read in individual dataframes for the portions. It isn't efficient, but it is readable.
    xyz_df = pd.read_csv(in_name, header=None, sep=",", skiprows=xyz_range)

    # Use ravel to stack the rows untop of one another
    stacked_xyz = pd.DataFrame(xyz_df.iloc[:, 1:4].values.ravel('F'))

    # Define the elements dataframe
    elements_df = pd.read_csv(in_name, header=None, sep=",", index_col=False, skiprows=element_range)

    with open(outname + ".case", 'w', newline="") as fout:
        # tells where to find the geometry (i.e. .geo file) and what format
        fout.write('FORMAT\ntype: ensight gold\n\nGEOMETRY\nmodel:                           ')
        # Scalars per node for 3d geometry
        fout.write(outname + ".geo\n")
        fout.close()
    with open(outname + '.geo', 'w', newline="") as fout:
        # Case header defining the name of the scalar array
        Title = str(outname) + ".inp"
        # fout to write the header
        fout.write("Title:  " + str(Title) + "\n")
        # fout to write the shape of the tetrahedral geometry (i.e. tetra4)
        fout.write('Description 2\nnode id given\nelement id given\npart\n         1\n')
        fout.write('Description PART\ncoordinates \n     ' + str(len(xyz_df)) + "\n")
        # use pandas in conjunction with fout to append the scalars
        xyz_df[0].to_csv(fout, header=False, index=False, sep=" ")
        # Write out the case format header with fout
        stacked_xyz.to_csv(fout, header=False, index=False, sep=" ")
        # Write out the elments
        fout.write("\ntria3\n" + str(len(elements_df)) + "\n")
        elements_df[0].to_csv(fout, header=False, index=False, sep=" ")
        elements_df.iloc[:, 1:].to_csv(fout, header=False, index=False, sep=" ")
        fout.close()

    #Write out the point cloud
    xyz_df.iloc[:, 1:4].to_csv(outname + "_pointcloud.csv", header=None, index=False)

    end = timer()
    end_time = (end - start)
    print(f"Coversion took {end_time:2.2f}\n")

def mesh_2d_to_mesh_3d(input_name):
    '''

    :param input_name:
    :return:
    '''
    start = timer()
    # Read in the mesh using trimesh
    mesh_2d = trimesh.load(str(input_name))

    #Print out the edge length information
    mesh_info(mesh_2d)

    #Redefine the input_name for writing
    input_name = input_name.replace(".ply", "")
    print("Coverting", input_name, "to 3d...")

    #Set up the save name and location for the inp file
    #Other file types available are Nastran (.bdf), Gmsh (.msh), Abaqus (*.inp), Diffpack (*.diff) and Inria Medit (*.mesh)
    save_location = input_name + "_3d.inp"

    #Use the gmsh interface to turn the 2d surface into a 3d volume and write out the inp file
    #(the_mesh, file_name=Save_name/location, max_element=Max_length_of_element, mesher_id=algortihm to use
    #1: Delaunay, 4: Frontal, 7: MMG3D, 10: HXT)
    trimesh.interfaces.gmsh.to_volume(mesh_2d, file_name=save_location, max_element=None, mesher_id=1)

    #test = trimesh.interfaces.gmsh.to_volume(mesh_2d, file_name=None, max_element=None, mesher_id=1)
    end = timer()
    end_time = (end - start)
    print("Coversion took", end_time, "\n")

def get_gmsh_info(gmsh_input):
    gmsh.initialize()
    gmsh.open(str(gmsh_input))
    dimensions = gmsh.model.getDimension()
    print("Info    : Mesh is {} dimensions.".format(dimensions))
    gmsh.finalize()
