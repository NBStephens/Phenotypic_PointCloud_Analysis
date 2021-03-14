import pathlib
from timeit import default_timer as timer

import regex as re


def _vtk_print_mesh_info(inputMesh):
    """
    Function to report basic information about a VTK mesh.
    :param inputMesh: Mesh file readable by pyvista
    :return:
    """
    if str(type(inputMesh)).split(".")[-1] == "vtkMultiBlockDataSet'>":
        print(
            f"Mesh has {inputMesh.GetNumberOfBlocks()} blocks...\n Information from first block only..."
        )
        inputMesh = inputMesh.GetBlock(0)
    else:
        pass
    x_min, x_max, y_min, y_max, z_min, z_max = inputMesh.GetBounds()
    center = inputMesh.GetCenter()
    cells = inputMesh.GetNumberOfCells()
    points = inputMesh.GetNumberOfPoints()
    cellType = inputMesh.GetMaxCellSize()
    # components = mesh.GetNumberOfPieces()
    if cellType == 3:
        meshType = "Triangular"
    elif cellType == 4:
        meshType = "Tetrahedral"
    else:
        meshType = str(int(cellType)) + " point"
    print(f"\nMesh type: {meshType} mesh")
    print(f"Elements:  {cells}, Nodes: {points}\n")
    print(f"Physcial size: x: {x_max:4.4f}, y: {y_max:4.4f}, z:{z_max:4.4f}")
    print(f"Origin:         {x_min:4.4f},     {y_min:4.4f},    {z_min:4.4f}")
    print(f"Center:          {center[0]:4.4f},    {center[1]:4.4f},   {center[2]:4.4f}")


def _end_timer(start_timer, message=""):
    """
    Simple function to print the end of a timer in a single line instead of being repeated.
    :param start_timer: timer start called using timer() after importing: from time import time as timer.
    :param message: String that makes the timing of what event more clear (e.g. "segmenting", "meshing").
    :return: Returns a sring mesuraing the end of a timed event in seconds.
    """
    start = start_timer
    message = str(message)
    end = timer()
    elapsed = abs(start - end)
    if message == "":
        print(f"Operation took: {float(elapsed):10.4f} seconds")
    else:
        print(f"{message} took: {float(elapsed):10.4f} seconds")


def _get_outDir(outDir):
    """
    Simple function to wrap an output directory using pathlib.
    :param outDir: Directory for writing out a file.
    :return:
    """
    if outDir == "":
        outDir = pathlib.Path.cwd()
    else:
        outDir = pathlib.Path(str(outDir))
    return outDir


def _get_inDir(inDir):
    """
    Simple function to wrap an input directory using pathlib.
    :param outDir: Directory for writing out a file.
    :return:
    """
    if inDir == "":
        inDir = pathlib.Path.cwd()
    else:
        inDir = pathlib.Path(str(inDir))
    return inDir


def get_output_path(directory, append_name=""):
    """
    Function to build a directory based upon the arguments passed in append. Takes a
    directory and uses pathlib to join that to the append passed.
    :param directory:     The input directory.
    :param kmeans_groups: a string to be appened to the pathlib object.
    :return:              Returns a new directory for output.
     example:
            Windows
                kmeans_groups =  2
                directory = pathlib.Path(r"C:\\Users\\nbs49\\Desktop\\ValidationSet_May26_tiff")
                output_path = get_output_path(directory, kmeans_groups)
            Linux/MacOS
                directory = pathlib.Path("/mnt/c/nbs49/Desktop/ValidationSet_May26_tiff")
                kmeans_groups =  2
                output_path = get_output_path(directory, kmeans_groups)
    """

    # Define the image output path for the segmented files.
    directory = pathlib.Path(directory)
    print(f"Working with directory {directory}")
    # The amount of kmeans groups used in the segmentation. Used to keep the files in distinct folders.
    append_name = str(append_name)
    # Define the output path by the initial directory and join (i.e. "+") the appropriate text.
    output_path = pathlib.Path(directory).joinpath(str(append_name))
    print(output_path)

    # Use pathlib to see if the output path exists, if it is there it returns True
    if pathlib.Path(output_path).exists() == False:

        # Prints a status method to the console using the format option, which fills in the {} with whatever
        # is in the ().
        print(f"\nOutput path {output_path} doesn't exist. Creating...\n\n")

        # Use pathlib to create the folder.
        pathlib.Path.mkdir(output_path)

    # Since it's a boolean return, and True is the only other option we will simply print the output.
    else:
        # This will print exactly what you tell it, including the space. The backslash n means new line.
        print(f"\nOutput path:\n               {output_path}\n\n")

    # Returns the output_path
    return output_path


def alpha_to_int(text):
    clean_text = int(text) if text.isdigit() else text
    return clean_text


def alpha_to_float(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [alpha_to_int(c) for c in re.split(r"(\d+)", text)]


def natural_keys_float(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    """
    return [
        alpha_to_float(c)
        for c in re.split(r"[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)", text)
    ]
