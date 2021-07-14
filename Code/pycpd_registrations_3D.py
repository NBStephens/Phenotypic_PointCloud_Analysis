"""
Script that uses pycpd to register point clouds using the coherent point drift algorithm
paper https://arxiv.org/pdf/0905.2635.pdf
An "average" point cloud with associated data is registered pairwise to a set of clouds
and then the data from the corresponding points mapped onto the "average" cloud through
linear interpolation. These can then be averaged across the points in relation to the
points.
The first rigid alignment doesn't always work if the clouds are poorly aligned to start
it is recommended that an external check is performed in cloud compare or a similar program.
Then the poorly aligned cloud can be manually aligned and saved for a second rigid step.
Following this an affine registration, and a deformable registration should be performed
and visually checked to make certain the correspondence makes sense.
Note that this script was written with the intent that the point clouds are created in
Paraview, and that the column data strcutures follow the scalar, x, y, z convention.
Note that dense point clouds will take up a great deal of memory, and it may be necessary
to reduce the amount of points used.
Author: Nick Stephens, Date: August, 2018
"""

# Print it out so we can see the progress
import os
import sys
from typing import Union

import vtk
import time
import glob
import pycpd
import socket
import shutil
import pathlib
import platform
import itertools
import numpy as np
import pandas as pd
import pyvista as pv
import seaborn as sns
from functools import partial
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from scipy.stats import variation
from mpl_toolkits.mplot3d import Axes3D
from vtk.util import numpy_support as npsup
from scipy.interpolate import griddata
from timeit import default_timer as timer
from typing import Union

script_dir = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(str(script_dir.parent))
from Code.PPCA_utils import get_output_path
from Code.PPCA_utils import _end_timer, _get_outDir, _vtk_print_mesh_info


def visualize(iteration, error, X, Y, ax):
    """
    Uses the callback in pycpd to visualize the registration of the two data sets in matplotlib.
    :param iteration: Number of iterations before the registration stops.
    :param error: Threshold for coorespondence between points before the registration stops.
    :param X: X axis of plot
    :param Y: Y axis of plot
    :param ax: axis object from matplotlib.
    """
    plt.cla()
    ax.scatter(
        X[:, 0], X[:, 1], X[:, 2], color="lime", label="Fixed", marker=".", alpha=0.5
    )  # Define the color of the points and their text
    ax.scatter(
        Y[:, 0],
        Y[:, 1],
        Y[:, 2],
        color="rebeccapurple",
        label="Moving",
        marker=".",
        alpha=0.5,
    )  # Define the color of the points and their text
    ax.text2D(
        0.87,
        0.92,
        "Iteration: {:d}\nError: {:06.4f}".format(iteration, error),
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        fontsize="large",
    )
    ax.legend(loc="upper left", fontsize="large", frameon=False)
    ax.grid(False)  # Gets rid of the grid lines
    ax.set_xticks([])  # Gets rid of the tick marks by creating an empty list
    ax.set_yticks([])  # Gets rid of the tick marks by creating an empty list
    ax.set_zticks([])  # Gets rid of the tick marks by creating an empty list
    ax.set_xlabel("X")  # Set the labels for the box
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # Change the panes to white
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # Change the panes to white
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  # Change the panes to white
    plt.draw()
    plt.pause(0.1)  # How often the image updates


def visual_update():
    """
    Uses the visualize callback for the registration.
    """
    fig = plt.figure()  # Start the plot
    ax = fig.add_subplot(111, projection="3d")  # Setup the 3d projection
    callback = partial(visualize, ax=ax)


# Callback to spit out the iteration in text form
def text_update(iteration, error, X, Y):
    """
    Uses the callback in pycpd to return a text based update of the registraion instead of the 3d plot.
    :param iteration: Number of iterations before the registration stops.
    :param error: Threshold for coorespondence between points before the registration stops.
    """
    print(f"\rIteration: {iteration}. Error: {error}.", end="")


# Prints out information about the point clouds
def print_points(moving, fixed):
    """
    Prints the number of point in the fixed and moving point clouds, along with their dimensionality.
    :param moving: A numpy array of the moving point cloud.
    :param fixed: A numpy array of the fixed point cloud.
    """
    print("Fixed nodes and dimensionality:", fixed.shape)
    print("Moving nodes and dimensionality:", moving.shape)


def print_moving_points(moving):
    """
    Prints the number of point in the moving point cloud, along with the dimensionality.
    :param moving: A numpy array of the moving point cloud.
    """
    print("Moving points", moving.shape[0], "Dimensionality:", moving.shape[1])
    print(moving)
    print("\n")


def print_fixed_points(fixed):
    """
    Prints the number of point in the fixed point cloud, along with the dimensionality.
    :param fixed: A numpy array of the fixed point cloud.
    """
    print("Moving points", fixed.shape[0], "Dimensionality:", fixed.shape[1])
    print(fixed)
    print("\n")


# Gets the correspondence between points for fixed
def get_fixed_correspondence(reg_P):
    """
    Produces a csv file showing the coorespondence between points and the probability that they are the same.
    This file can be very large, but is worthwhile for verifying the algorithm is doing what you want it to do.
    :param reg_P: An object containing a matrix of the values returned by pycpd between iterations.
    :param idVect: The index/identity of the points in the point cloud.
    :param maxVect: The maximum probabilty along the rows for each point.
    :param correspondence_points: A stack of the idVect and maxVect.
    """
    correspondence = reg_P  # Store the probability matrix
    idVect = np.argmax(
        correspondence, axis=0
    )  # Grab the id of the max values in the matrix across rows
    maxVect = np.max(correspondence, axis=0)  # Grab the max values in those vectors
    correspondence_points = np.vstack((idVect, maxVect))  # Put the two together
    fixed_correspondence = pd.DataFrame(
        correspondence_points
    )  # Place them into a datframe with the fixed index along the column
    return fixed_correspondence


# Gets the correspondence between points for moving
def get_moving_correspondence(reg_P):
    """
    Produces a csv file showing the coorespondence between points and the probability that they are the same.
    This file can be very large, but is worthwhile for verifying the algorithm is doing what you want it to do.
    :param reg_P: An object containing a matrix of the values returned by pycpd between iterations.
    """
    correspondence = reg_P
    movingidVect = np.argmax(correspondence, axis=1)
    movingmaxVect = np.max(correspondence, axis=1)
    moving_correspondence_points = np.vstack((movingidVect, movingmaxVect))
    moving_correspondence = pd.DataFrame(moving_correspondence_points)
    return moving_correspondence


def get_distance(xyz1, xyz2):
    """
    Gets the distance between z,yz, coordinates.
    distance = √(x2−x1)2+(y2−y1)2+(z2−z1)2
    https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
    example:
    a = pd.read_csv("SSF_7_C1_Whole_BVTV_point_cloud_1.75mm0_rigid_moving.csv", header=None)
    b = pd.read_csv("SSF_7_C1_Whole_BVTV_point_cloud_1.75mm0_deformable_moving.csv", header=None)
    c['Distance'] = pd.DataFrame(sqrt_einsum(a, b))
    :param xyz1: first pandas dataframe or numpy array with xyz coordinates
    :param xyz2: second pandas dataframe or numpy array with xyz coordinates
    :return: returns the distance between two points
    """
    a_minus_b = xyz1 - xyz2
    return np.sqrt(np.einsum("ij,ij->i", a_minus_b, a_minus_b))


def get_centroid(point_cloud):
    arr = point_cloud
    length = point_cloud.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    sum_z = np.sum(arr[:, 2])
    return np.array([sum_x / length, sum_y / length, sum_z / length])


# Define the rigid transformation from pycpd
def visual_rigid(name_in, moving, fixed, iterations, tolerance):
    """
    :param name_in:
    :param moving:
    :param fixed:
    :param iterations:
    :param tolerance:
    :return:
    """
    print_points(moving, fixed)
    visual_update()
    reg = pycpd.RigidRegistration(
        **{"X": fixed, "Y": moving}
    )  # Read in the moving and fixed
    reg.max_iterations = iterations  # Define the number of iterations
    reg.tolerance = tolerance  # Define the tolerance to stop at
    start = timer()  # Time the process
    reg.register(callback)  # to visualize the registration
    plt.show()  # Bring the plot up
    print(Y)  # Print the old matrix
    print("\n")
    print("New matrix:")
    print(reg.TY)  # The registered matrix
    # np.savetxt(name_in + "rigid_fixed.csv", reg.XX, delimiter=",", fmt='%f') #Save the points to a csv without scientific notation
    np.savetxt(
        name_in + "rigid_moving.csv", reg.TY, delimiter=",", fmt="%f"
    )  # Same with the rigid
    if matrix == "False":
        pass
    else:
        np.savetxt(
            name_in + "_rigid_correspondence_matrix.csv", reg.P, delimiter=",", fmt="%f"
        )
    df = get_fixed_correspondence(reg.P)
    df.to_csv(
        name_in + "rigid_points_correspondence.csv", sep=",", index=False
    )  # And then the corresponding points with probability
    end = timer()
    end_time = end - start
    print(end_time)


def visual_affine(name_in, moving, fixed, iterations, tolerance, matrix=False):
    """
    :param name_in:
    :param moving:
    :param fixed:
    :param iterations:
    :param tolerance:
    :param matrix:
    :return:
    """
    print_points(moving, fixed)
    visual_update()
    reg = pycpd.AffineRegistration(**{"X": fixed, "Y": moving})
    reg.max_iterations = iterations
    reg.tolerance = tolerance
    reg.register(callback)
    start = timer()
    plt.show()
    print(Y)
    print("\n")
    print("New matrix:")
    print(reg.TY)  # The registered matrix
    np.savetxt(name_in + "_affine_moving.csv", reg.TY, delimiter=",", fmt="%f")
    if matrix == False:
        pass
    else:
        np.savetxt(
            name_in + "_affine_correspondence_matrix.csv",
            reg.P,
            delimiter=",",
            fmt="%f",
        )
    df = get_fixed_correspondence(reg.P)
    df.to_csv(name_in + "_affine_points_correspondence.csv", sep=",", index=False)
    end = timer()
    end_time = end - start
    print(end_time)


def visual_deformable(name_in, moving, fixed, iterations, tolerance, matrix=False):
    """
    :param name_in:
    :param moving:
    :param fixed:
    :param iterations:
    :param tolerance:
    :param matrix:
    :return:
    """
    print_points(moving, fixed)
    visual_update()
    reg = pycpd.DeformableRegistration(**{"X": fixed, "Y": moving})
    reg.max_iterations = iterations
    reg.tolerance = tolerance
    reg.register(callback)
    start = timer()
    plt.show()
    print(moving)
    print("\n")
    print("New matrix:")
    print(reg.TY)  # The registered matrix
    np.savetxt(name_in + "deformable_moving.csv", reg.TY, delimiter=",", fmt="%f")
    if matrix == False:
        pass
    else:
        np.savetxt(
            name_in + "deformable_correspondence_matrix.csv",
            reg.P,
            delimiter=",",
            fmt="%f",
        )
    df = get_fixed_correspondence(reg.P)
    df.to_csv(name_in + "deformable_points_correspondence.csv", sep=",", index=False)
    end = timer()
    end_time = end - start
    print(end_time)


def get_transform(name_in, reg, tranform_type, outDir=""):
    """
    Function to write out the transofrmation from pycpd
    :param name_in:
    :param reg:
    :param tranform_type:
    :return:
    """
    name_in = str(name_in)
    df = get_fixed_correspondence(reg.P)
    tranform_type = str(tranform_type)

    transform_name = f"{name_in}_{str(tranform_type)}_transformations.csv"
    correspondence_name = f"{name_in}_{str(tranform_type)}_points_correspondence.csv"

    if outDir != "":
        transform_dir = pathlib.Path(outDir).joinpath(str(tranform_type))
        transform_name = transform_dir.joinpath(transform_name)
        correspondence_name = transform_dir.joinpath(correspondence_name)

    if tranform_type == "deformable":
        pass

    elif tranform_type == "rigid":
        reg_matrix = pd.DataFrame(reg.R)
        trans_matrix = pd.DataFrame(reg.t)
        transformations = pd.concat([reg_matrix, trans_matrix], axis=1)
        transformations.columns = ["X-axis", "Y-axis", "Z-axis", "Origin"]
        transformations["Scale"] = reg.s
        transformations.to_csv(str(transform_name), sep=",", index=False)

    elif tranform_type == "affine":
        reg_matrix = pd.DataFrame(reg.B)
        trans_matrix = pd.DataFrame(reg.t)
        transformations = pd.concat([reg_matrix, trans_matrix], axis=1)
        transformations.columns = ["X-axis", "Y-axis", "Z-axis", "Origin"]
        transformations["Scale"] = 0.0
        transformations.to_csv(str(transform_name), sep=",", index=False)
    else:
        pass
    df.to_csv(str(correspondence_name), sep=",", index=False)


def rigid_low_to_high(
    name_in, moving, fixed, iterations, tolerance, matrix=False, outDir=""
):
    """

    :param name_in:
    :param moving:
    :param fixed:
    :param iterations:
    :param tolerance:
    :param matrix:
    :return:
    """

    outname = f"{name_in}_aligned_original.csv"
    matrixname = f"{name_in}_aligned_original_correspondence_matrix.csv"

    if outDir != "":
        save_dir = pathlib.Path(outDir).joinpath("rigid")
        outname = save_dir.joinpath(outname)
        matrixname = save_dir.joinpath(matrixname)

    print_points(moving, fixed)
    reg = pycpd.RigidRegistration(**{"X": fixed, "Y": moving})
    reg.max_iterations = iterations
    reg.tolerance = tolerance
    reg.register(text_update)
    start = timer()
    print(moving)
    print("\n")
    print("New matrix:")
    print(reg.TY)

    np.savetxt(str(outname), reg.TY, delimiter=",", fmt="%f")

    if matrix != False:
        np.savetxt(str(matrixname), reg.P, delimiter=",", fmt="%f")

    get_transform(name_in, reg, "rigid", outDir=outDir)
    _end_timer(start_timer=start, message="High to low registration")


# These use the text callback which is better for looping over a list
def rigid(name_in, moving, fixed, iterations, tolerance, matrix=False, outDir=""):
    """

    :param name_in:
    :param moving:
    :param fixed:
    :param iterations:
    :param tolerance:
    :param matrix:
    :return:
    """

    transform_type = "rigid"
    outname = f"{name_in}_{transform_type}_moving.csv"
    matrixname = f"{name_in}_{transform_type}_correspondence_matrix.csv"

    if outDir != "":
        save_dir = pathlib.Path(outDir).joinpath(transform_type)
        outname = save_dir.joinpath(outname)
        matrixname = save_dir.joinpath(matrixname)

    print_points(moving, fixed)
    reg = pycpd.RigidRegistration(
        **{"X": fixed, "Y": moving}
    )  # Read in the moving and fixed
    reg.max_iterations = iterations
    reg.tolerance = tolerance
    reg.register(text_update)
    start = timer()
    print(moving)
    print("\n")
    print("New matrix:")
    print(reg.TY)  # The registered matrix

    np.savetxt(str(outname), reg.TY, delimiter=",", fmt="%f")

    if matrix != False:
        np.savetxt(str(matrixname), reg.P, delimiter=",", fmt="%f")

    get_transform(name_in, reg, tranform_type=transform_type, outDir=outDir)
    _end_timer(start_timer=start, message=f"{transform_type} registration")


def affine(name_in, moving, fixed, iterations, tolerance, matrix=False, outDir=""):
    """
    :param name_in:
    :param moving:
    :param fixed:
    :param iterations:
    :param tolerance:
    :param matrix:
    :return:
    """

    transform_type = "affine"
    outname = f"{name_in}_{transform_type}_moving.csv"
    matrixname = f"{name_in}_{transform_type}_correspondence_matrix.csv"

    if outDir != "":
        save_dir = pathlib.Path(outDir).joinpath(transform_type)
        outname = save_dir.joinpath(outname)
        matrixname = save_dir.joinpath(matrixname)

    print_points(moving, fixed)
    reg = pycpd.AffineRegistration(**{"X": fixed, "Y": moving})
    reg.max_iterations = iterations
    reg.tolerance = tolerance
    reg.register(text_update)
    start = timer()
    print(moving)
    print("\n")
    print("New matrix:")
    print(reg.TY)  # The registered matrix
    np.savetxt(str(outname), reg.TY, delimiter=",", fmt="%f")

    if matrix != False:
        np.savetxt(str(matrixname), reg.P, delimiter=",", fmt="%f")

    get_transform(name_in, reg, tranform_type=transform_type, outDir=outDir)
    _end_timer(start_timer=start, message=f"{transform_type} registration")


def deformable(name_in, moving, fixed, iterations, tolerance, matrix=False, outDir=""):
    """
    :param name_in:
    :param moving:
    :param fixed:
    :param iterations:
    :param tolerance:
    :param matrix:
    :return:
    """

    transform_type = "deformable"
    outname = f"{name_in}_{transform_type}_moving.csv"
    matrixname = f"{name_in}_{transform_type}_correspondence_matrix.csv"

    if outDir != "":
        save_dir = pathlib.Path(outDir).joinpath(transform_type)
        outname = save_dir.joinpath(outname)
        matrixname = save_dir.joinpath(matrixname)

    print_points(moving, fixed)
    reg = pycpd.DeformableRegistration(**{"X": fixed, "Y": moving})
    reg.max_iterations = iterations
    reg.tolerance = tolerance
    reg.register(text_update)
    start = timer()
    print(moving)
    print("\n")
    print("New matrix:")
    print(reg.TY)  # The registered matrix

    np.savetxt(str(outname), reg.TY, delimiter=",", fmt="%f")

    if matrix != False:
        np.savetxt(str(matrixname), reg.P, delimiter=",", fmt="%f")

    get_transform(name_in, reg, tranform_type=transform_type, outDir=outDir)
    _end_timer(start_timer=start, message=f"{transform_type} registration")


# Uses scipy interpolate to map the original values to the registered points
def map_cloud(name_in, grid, points, original, canonical_geo, scalar, scalar_values):
    """
    :param name_in:
    :param grid:
    :param values:
    :param points:
    :param original:
    :param canonical_geo:
    :return:
    """
    scalar = str(scalar)
    scalar_values.columns = [str(scalar)]
    print("Number of points to map:", scalar_values.shape[0])
    # Define the points and then the grid
    points = points.values
    grid = grid.values
    print("Grid length", len(grid))
    # Check to make share there are the same dimensions
    if points.shape[0] - scalar_values.shape[0] != 0:
        print("Points don't match with values")
    else:
        print("Mapping", points.shape[0], "values to", grid.shape[0], "nodes.")
        # Place the values into a matrix
        scalar_values = scalar_values.values
        # There are other interpolation methods, but since it is spatial we use nearest
        mapped = griddata(points, scalar_values, grid, method="nearest")
        print(str(scalar) + " mapped!")
        # Put spatial values into a pandas dataframe
        cloud_coord = pd.DataFrame(grid)
        # Asign a header to the dataframe
        cloud_coord.columns = ["x", "y", "x"]
        # Place the interpolated values into a dataframe
        scalar_mapped = pd.DataFrame(mapped)
        print(scalar_mapped)
        # Write out the values to a ensight gold scalar format for paraview
        print("Writing out case file...\n")
        with open(name_in + "_" + str(scalar) + "_original_mapped.Esca1", "w") as fout:
            col_name = "ESca1" + name_in + "_" + str(scalar) + "_original_mapped"
            fout.write(col_name)
            fout.write("\npart\n         1\ntetra4\n")
            scalar_mapped.to_csv(
                fout, header=False, index=False, sep=" ", line_terminator="\n"
            )
        with open(name_in + "_" + str(scalar) + "_original_mapped.case", "w") as fout:
            col_name = name_in
            fout.write(
                "FORMAT\ntype: ensight gold\n\nGEOMETRY\nmodel:                           "
            )
            fout.write(
                col_name + "_original_mapped.geo\n\nVARIABLE\nscalar per node:  "
            )
            fout.write(
                "ESca1"
                + col_name
                + "_original_mapped"
                + "  "
                + name_in
                + "_"
                + str(scalar)
                + "_original_mapped.Esca1"
            )
        # Copy the connectivity from the original case file, which must be ascii and not binary
        shutil.copy(canonical_geo, col_name + "_original_mapped.geo")
        # place a column header onto the dataframe and then concat it to the original coordinates
        scalar_mapped.columns = [str(scalar)]
        mapped_cloud = pd.concat([cloud_coord, scalar_mapped], axis=1)
        # Write out the CSV file so you can view it
        mapped_cloud.to_csv(
            name_in + "_" + str(scalar) + "_deformed_mapped.csv", sep=",", index=False
        )
        # Read in the original, unregistered point cloud, and concat the interpolated values
        original = pd.read_csv(original, sep=",", header=None)
        original = original.iloc[:, 0:4]
        original.columns = ["x", "y", "z"]
        original_mapped = pd.concat([original, scalar_mapped], axis=1)
        # Write out the undistorted point cloud with the interpolated values
        original_mapped.to_csv(
            name_in + "_" + str(scalar) + "_original_mapped.csv", index=False
        )


# Uses scipy interpolate to map the original values to the registered points
def map_cloud_max_normalized(
    name_in, grid, points, original, canonical_geo, scalar, scalar_values
):
    """
    Function to divide scalar by the max value in the scalar, to normalize between 0:1. Adapted from Saers 2019b (https://doi.org/10.1016/j.jhevol.2019.102654).
    :param name_in:
    :param grid:
    :param values:
    :param points:
    :param original:
    :param canonical_geo:
    :return:
    """
    scalar = str(scalar)
    scalar_values.columns = [str(scalar)]
    print("Number of points to map:", scalar_values.shape[0])
    # Define the points and then the grid
    points = points.values
    grid = grid.values
    print("Grid length", len(grid))
    # Check to make share there are the same dimensions
    if points.shape[0] - scalar_values.shape[0] != 0:
        print("Points don't match with values")
    else:
        print("Mapping", points.shape[0], "values to", grid.shape[0], "nodes.")
        # Place the values into a matrix
        scalar_values = scalar_values.values
        max_value = scalar_values.max()
        print(max_value)
        scalar_values = scalar_values / max_value
        # There are other interpolation methods, but since it is spatial we use nearest
        mapped = griddata(points, scalar_values, grid, method="nearest")
        print(str(scalar) + " mapped!")
        # Put spatial values into a pandas dataframe
        cloud_coord = pd.DataFrame(grid)
        # Asign a header to the dataframe
        cloud_coord.columns = ["x", "y", "x"]
        # Place the interpolated values into a dataframe
        scalar_mapped = pd.DataFrame(mapped)
        print(scalar_mapped)
        # Write out the values to a ensight gold scalar format for paraview
        print("Writing out _max_normalized case file...\n")
        with open(
            name_in + "_" + str(scalar) + "_original_mapped_max_normalized.Esca1", "w"
        ) as fout:
            col_name = (
                "ESca1"
                + name_in
                + "_"
                + str(scalar)
                + "_original_mapped_max_normalized"
            )
            fout.write(col_name)
            fout.write("\npart\n         1\ntetra4\n")
            scalar_mapped.to_csv(
                fout, header=False, index=False, sep=" ", line_terminator="\n"
            )
        with open(
            name_in + "_" + str(scalar) + "_original_mapped_max_normalized.case", "w"
        ) as fout:
            col_name = name_in
            fout.write(
                "FORMAT\ntype: ensight gold\n\nGEOMETRY\nmodel:                           "
            )
            fout.write(
                col_name + "_original_mapped.geo\n\nVARIABLE\nscalar per node:  "
            )
            fout.write(
                "ESca1"
                + col_name
                + "_original_mapped_max_normalized"
                + "  "
                + name_in
                + "_"
                + str(scalar)
                + "_original_mapped_max_normalized.Esca1"
            )
        # Copy the connectivity from the original case file, which must be ascii and not binary
        shutil.copy(canonical_geo, col_name + "_original_mapped.geo")
        # place a column header onto the dataframe and then concat it to the original coordinates
        scalar_mapped.columns = [str(scalar)]
        mapped_cloud = pd.concat([cloud_coord, scalar_mapped], axis=1)
        # Write out the CSV file so you can view it
        mapped_cloud.to_csv(
            name_in + "_" + str(scalar) + "_deformed_mapped_max_normalized.csv",
            sep=",",
            index=False,
        )
        # Read in the original, unregistered point cloud, and concat the interpolated values
        original = pd.read_csv(original, sep=",", header=None)
        original = original.iloc[:, 0:4]
        original.columns = ["x", "y", "z"]
        original_mapped = pd.concat([original, scalar_mapped], axis=1)
        # Write out the undistorted point cloud with the interpolated values
        original_mapped.to_csv(
            name_in + "_" + str(scalar) + "_original_mapped_max_normalized.csv",
            index=False,
        )


def write_case(outname, scalars, scalar_type, canonical_geo):
    """
    :param outname: The output name of the case file.
    :param scalars: A single row from a pandas dataframe or series containign scalar values
    :param scalar_type: A string containing what the scalar is (e.g. BVTV, DA, p-values etc.)
    :param canonical_geo: A .geo file matching the morphology of the scalar infromation.
    :return: Writes out a .case file that can then be viewed in paraview.
    """
    scalar_type = str(scalar_type)
    outname = str(outname) + "_" + str(scalar_type)
    scalars = scalars
    canonical_geo = str(canonical_geo)

    # Open an Ensight Gold .case scalar text file to be written. The newline forces the windows newline type.
    with open("ESca1" + outname + ".Esca1", "w", newline="") as fout:
        # Define the column name so the files all reference one another
        column_name = "ESca1" + outname
        fout.write(column_name)

        # Write the first two lines, denoting a tetrahedral grid.
        fout.write("\npart\n         1\ntetra4\n")

        # The scalars are then written line by line using fout
        scalars.to_csv(fout, header=False, index=False, sep=" ", float_format="%.10f")
    # Open an Ensight Gold .case header
    with open(outname + ".case", "w", newline="") as fout:
        fout.write(
            "FORMAT\ntype: ensight gold\n\nGEOMETRY\nmodel:                           "
        )
        fout.write(outname + ".geo\n\nVARIABLE\nscalar per node:  ")
        fout.write(column_name + " " + "  " + column_name + ".Esca1")
        shutil.copy(canonical_geo, outname + ".geo")


# Collect BV/TV or DA values together and out a mean case, a standard deviation case, and one
# with the coefficient of variation.
# Takes in a file list to concactenate csv, or txt files together.
# Uses the outname to rename all the files consistently.
# The canonical geo copies the ensight gold case geometry and renames accordingly
def get_mean_case(mean_fileList, outname, canonical_geo, scalar, outDir=""):
    """
    :param mean_fileList:
    :param outname:
    :param canonical_geo:
    :return:
    """
    # Open an empty list to be appeneded to
    np_array_list = []
    scalar = str(scalar)
    if outDir == "":
        outDir = pathlib.Path.cwd()
    else:
        outDir = pathlib.Path(outDir)
    # Loop throug the file_list and read in the values

    currentDir = pathlib.Path.cwd()

    shutil.copy(
        canonical_geo, pathlib.Path(outDir).joinpath(outname + "_original_average.geo")
    )
    for f in mean_fileList:
        # Use pandas to read the comma separated values
        data = pd.read_csv(f, sep=",")
        # Select the scalar column and push it to a numpy array
        data = data.iloc[:, 3].values
        print(len(data))
        # Append to the list defined above
        np_array_list.append(data)
        # Get the name from the loop
        name = f
        # Replace the "csv" string so it can be appended at the end
        name = name.replace(".csv", "")
        print(name)
        # Use numpy save text to write out the indiviedual values
        outputName = pathlib.Path(outDir).joinpath(name + "_" + str(scalar) + ".csv")
        np.savetxt(str(outputName), data, "%5.10f", delimiter=",")
    # Make sure we have a string for the output name
    outname = str(outname)
    # Stack the numpy arrays vertically (which changed from the last version)
    comb_np_array = np.vstack(np_array_list)
    # Convert this to a dataframe, and transpose it so the individual files are column-wise.
    scalar_array = pd.DataFrame(comb_np_array).T

    # Write out the matrix to a csv
    scalaroutputName = pathlib.Path(outDir).joinpath(
        outname + "_all_" + str(scalar) + "_values.csv"
    )
    scalar_array.to_csv(str(scalaroutputName), index=False)

    # Create a new dataframe with pandas and get the mean across the rows
    mean_scalar = pd.DataFrame(scalar_array.mean(axis=1, numeric_only=True))

    # Check to make sure the vertex values are the same
    print("number of vertex values:", mean_scalar.shape[0])
    # Write out the mean to a csv
    mean_scalaroutputName = pathlib.Path(outDir).joinpath(
        outname + "mean_" + str(scalar) + "_3d.csv"
    )
    mean_scalar.to_csv(str(mean_scalaroutputName), index=False, sep=" ")  # pd.mean_bin
    os.chdir(outDir)
    # Write the mean values to a case file, with the fout in ensigh gold format
    # the "newline" argument prevents an automatic return being written on windows
    with open("ESca1" + outname + "_original_average.Esca1", "w", newline="") as fout:
        # Case header defining the name of the scalar array
        col_name1 = "ESca1" + outname + "_original_average"
        # fout to write the header
        fout.write(col_name1)
        # fout to write the shape of the tetrahedral geometry (i.e. tetra4)
        fout.write("\npart\n         1\ntetra4\n")
        # use pandas in conjunction with fout to append the scalars
        mean_scalar.to_csv(fout, header=False, index=False, sep=" ")
        # Write out the case format header with fout
    with open(outname + "_original_average.case", "w", newline="") as fout:
        # tells where to find the geometry (i.e. .geo file) and what format
        fout.write(
            "FORMAT\ntype: ensight gold\n\nGEOMETRY\nmodel:                           "
        )
        # Scalars per node for 3d geometry
        fout.write(outname + "_original_average.geo\n\nVARIABLE\nscalar per node:  ")
        # Case header information for the name of the scalar file
        fout.write(col_name1 + " " + "  " + col_name1 + ".Esca1")

    # Calculatel the standard deviation row wise and write out the case
    standard_dev = scalar_array.std(axis=1)
    print(standard_dev)
    with open(
        "ESca1" + outname + "_original_standard_dev.Esca1", "w", newline=""
    ) as fout:
        col_name2 = "ESca1" + outname + "_original_standard_dev"
        fout.write(col_name2)
        fout.write("\npart\n         1\ntetra4\n")
        standard_dev.to_csv(fout, header=False, index=False, sep=" ")
    with open(outname + "_original_standard_dev.case", "w", newline="") as fout:
        fout.write(
            "FORMAT\ntype: ensight gold\n\nGEOMETRY\nmodel:                           "
        )
        fout.write(outname + "_original_average.geo\n\nVARIABLE\nscalar per node:  ")
        fout.write(col_name2 + " " + "  " + col_name2 + ".Esca1")
    # Calculate the coefficient of variation and write the case
    coef_var = scalar_array.std(axis=1) / scalar_array.mean(axis=1)
    with open("ESca1" + outname + "_original_coef_var.Esca1", "w", newline="") as fout:
        col_name3 = "ESca1" + outname + "_original_coef_var"
        fout.write(col_name3)
        fout.write("\npart\n         1\ntetra4\n")
        coef_var.to_csv(fout, header=False, index=False, sep=" ")
    with open(outname + "_original_coef_var.case", "w", newline="") as fout:
        fout.write(
            "FORMAT\ntype: ensight gold\n\nGEOMETRY\nmodel:                           "
        )
        fout.write(outname + "_original_average.geo\n\nVARIABLE\nscalar per node:  ")
        fout.write(col_name3 + " " + "  " + col_name3 + ".Esca1")
    # Copy and rename the geometry from the original case file.
    # Note that this muyst be ascii, not binary as output by avizo
    os.chdir(currentDir)


# Function for the initial rigid rotation of points
def initial_rigid(
    name_in,
    moving,
    auto3d_dir="",
    auto3d_ref="trabecular",
    rotation_matrix="",
    origin="",
    outdir="",
):
    """
    :param name_in: name of point cloud file to be transformed
    :param moving: point cloud x,y,z coordinates
    :param auto3d_dir: location of auto3dGM results directory with the roation and center
    :param rotation_matrix: 3x3 numpy array to apply to x,y,z coordinates
    :param origin: optional input of a 1x3 numpy array to apply to the x,y,z coordinates for translation
    :return: Returns new point cloud with the transformed x,y,z coordinates
    """
    # Start the timer for the operation
    start = timer()
    # If rotation matrix is left blank then grab the results from auto3dgm
    if rotation_matrix == "":
        # Can't have a blank directory if no rotation matrix provided
        if auto3d_dir == "":
            print(
                "If the rotation matrix isn't provided, you must include an auto3dgm directory!"
            )
        else:
            # Read in the auto3dgm directory to pull the rotation matrix
            auto3d_dir = pathlib.Path(auto3d_dir)
            print(f"\n\n{auto3d_dir}\n\n")
            rotation_file = pathlib.Path(auto3d_dir).joinpath(
                f"{name_in}_{str(auto3d_ref)}_rotation_matrix.txt"
            )
            rotation_matrix = pd.read_csv(
                rotation_file, header=None, delim_whitespace=True
            )
            # Check to make sure it's the right shape
            print(rotation_matrix.shape)
            try:
                if rotation_matrix.shape[0] == 4:
                    rotation_matrix = pd.read_csv(rotation_file, sep=",")
            except AttributeError:
                rotation_matrix = pd.read_csv(rotation_file, sep=",")

            if rotation_matrix.shape[1] != moving.shape[1]:
                print(
                    "Matrix and xyz points do not match! \n Must be a nx3 and 3x3 matrix!"
                )
                print("Rotation matrix:\n", rotation_matrix)
            else:
                print("Rotation matrix:\n", rotation_matrix, "\n")
    else:
        if rotation_matrix.shape[1] != moving.shape[1]:
            print(
                "Matrix provided, but the xyz points do not match! \n Must be a 3x3 matrix!"
            )
            print("Rotation matrix:\n", rotation_matrix)
            print(rotation_matrix.shape, "\n")
        else:
            print("Rotation matrix:\n", rotation_matrix, "\n")
    if origin == "":
        # Get the origin
        try:
            origin = pathlib.Path(auto3d_dir).joinpath(
                str(name_in) + "_" + str(auto3d_ref) + "_center.txt"
            )
            origin = pd.read_csv(
                origin, header=None, delim_whitespace=True
            ).values.flatten()
        except:
            origin = pathlib.Path(auto3d_dir).joinpath(
                str(name_in) + "_" + str(auto3d_ref) + "_center.csv"
            )
            origin = pd.read_csv(origin, sep=",").values.flatten()

        if origin.shape[0] != 3:
            print("Origin is not in the right order! \n Must be a 1x3 matrix!")
            print(origin)
            origin = np.array[0.0, 0.0, 0.0]
            print("Not applying translation")
        else:
            print("origin is:\n", origin)
    else:
        if origin.shape[0] != 3:
            print("Origin is not in the right order! \n Must be a 1x3 matrix!")
            print(origin)
            origin = np.array[0.0, 0.0, 0.0]
            print("Not applying translation")
        else:
            print("origin is:\n", origin)
    # Check to see if the determinant of the rotation is negative.
    if np.linalg.det(rotation_matrix) < 0:
        print("Determinant is negative and the canonical point cloud will be mirrored.")
    else:
        pass
    print_points(moving, rotation_matrix)
    # Reg is the matrix multiplication (@) of the points and the transform
    reg = (rotation_matrix @ moving.T).T
    reg = origin + reg
    print(moving)
    print("\n New matrix:")
    print(reg)
    if outdir == "":
        np.savetxt(name_in + "_aligned_original.csv", reg, delimiter=",", fmt="%f")
    else:
        outdir = pathlib.Path(outdir)
        outName = outdir.joinpath(f"{name_in}_aligned_original.csv")
        np.savetxt(str(outName), reg, delimiter=",", fmt="%f")
    end = timer()
    end_time = end - start
    print(end_time, "\n")


def set_origin_to_zero(name_In, point_cloud, outDir=""):
    name_in = name_In
    print(point_cloud)
    print(point_cloud.shape)
    centroid = get_centroid(point_cloud)
    print(f"Centroid: {centroid}\n")
    print("Setting centroid origin to ~0")
    centered = point_cloud - centroid
    print("\ncentroid:")
    new_centroid = get_centroid(centered)
    with np.printoptions(precision=3, suppress=True):
        print(new_centroid)
    if outDir == "":
        np.savetxt(name_in + "_aligned_original.csv", centered, delimiter=",", fmt="%f")
    else:
        outDir = pathlib.Path(outDir)
        save_name = outDir.joinpath(f"{name_in}_aligned_original.csv")
        np.savetxt(str(save_name), centered, delimiter=",", fmt="%f")


def _get_mean_case_groups(group_list, group_identifiers, bone, canonical_geo):
    if type(canonical_geo) == list:
        canonical_geo = canonical_geo[0]
    dictionary = dict(zip(group_list, group_identifiers))
    print(dictionary)
    for (key, values) in dictionary.items():
        group = list(values)
        group_list_bvtv = []
        group_list_da = []
        for i in group:
            try:
                group_list_bvtv.extend(
                    glob.glob("*" + str(i) + "*BVTV_original_mapped.csv")
                )
            except:
                print("No members for BVTV {} group\n.".format(str(key)))

            try:
                group_list_da.extend(
                    glob.glob("*" + str(i) + "*DA_original_mapped.csv")
                )
            except:
                print("No members for DA {} group\n.".format(str(key)))

        print("\n{} BVTV list contains {} items".format(key, len(group_list_bvtv)))

        if len(group_list_bvtv) > 1:
            outname = "Mean_" + str(key) + "_" + str(bone) + "_BVTV"
            print(outname)
            get_mean_case(group_list_bvtv, str(outname), canonical_geo, "BVTV")
        else:
            print("Not enough individuals to create a mean.\n")

        print("\n{} DA list contains {} items".format(key, len(group_list_da)))

        if len(group_list_da) > 1:
            outname = "Mean_" + str(key) + "_" + str(bone) + "_DA"
            get_mean_case(group_list_bvtv, str(outname), canonical_geo, "DA")

        else:
            print("Not enough individuals to create a mean.\n")


def get_mean_case_groups(
    group_list,
    group_identifiers,
    bone,
    canonical_geo,
    scalars=["BVTV", "DA", "CtTh"],
    max_normalized=True,
):

    if type(canonical_geo) == list:
        canonical_geo = canonical_geo[0]

    dictionary = dict(zip(group_list, group_identifiers))
    print(dictionary)

    for scales in scalars:
        print(f"Averaging {scales}")
        for key, values in dictionary.items():
            if type(values) == str:
                group = values
                try:
                    group_scalars = glob.glob(
                        "*"
                        + str(group)
                        + "*"
                        + str(bone)
                        + "*"
                        + str(scales)
                        + "_original_mapped.csv"
                    )
                    if max_normalized == True:
                        group_scalars_max = glob.glob(
                            "*"
                            + str(group)
                            + "*"
                            + str(bone)
                            + "*"
                            + str(scales)
                            + "_original_max_normalized.csv"
                        )
                except:
                    print(f"No members for {scales} {group} group\n.")
            elif type(values) == list:
                group_scalars = []
                group = list(values)
                for i in group:
                    try:
                        group_scalars.extend(
                            glob.glob(
                                "*"
                                + str(i)
                                + "*"
                                + str(bone)
                                + "*"
                                + str(scales)
                                + "_original_mapped.csv"
                            )
                        )
                        if max_normalized == True:
                            group_scalars_max.extend(
                                glob.glob(
                                    "*"
                                    + str(i)
                                    + "*"
                                    + str(bone)
                                    + "*"
                                    + str(scales)
                                    + "_original_mapped_max_normalized.csv"
                                )
                            )
                    except:
                        print(f"No members for {scales} {group} group\n.")
            else:
                print("Group identifiers not understood....")

            if len(group_scalars) > 1:
                print(f"\n{key} {scales} list contains {len(group_scalars)} items")
                get_output_path(directory=pathlib.Path.cwd(), append_name=str(scales))
                outPath = pathlib.Path.cwd().joinpath(str(scales))
                outname = "Mean_" + str(key) + "_" + str(bone) + "_" + str(scales)
                get_mean_case(
                    group_scalars,
                    str(outname),
                    canonical_geo=canonical_geo,
                    scalar=str(scales),
                    outDir=outPath,
                )
            else:
                print(f"Not enough individuals in {key} to create a mean {scales}.\n")
                if max_normalized == True:
                    outname_max = (
                        "Mean_"
                        + str(key)
                        + "_"
                        + str(bone)
                        + "_"
                        + str(scales)
                        + "max_normalized"
                    )
                    if len(group_scalars_max) > 1:
                        get_mean_case(
                            group_scalars_max,
                            str(outname_max),
                            canonical_geo=canonical_geo,
                            scalar=str(scales),
                            outDir=outPath,
                        )
                else:
                    print(
                        f"Not enough individuals in {key} to create a mean for max normalized {scales}.\n"
                    )


def interpolation_comparison(name_in, scalarsOriginal, scalarsInterpolated, scalar):
    """
    Diagnostic function to get the means before and after mapping, to check for major differences.
    :param name_in: String of the base name, which will become the column name for a dataframe header.
    :param scalarsOriginal: The original scalars, can either be a string or the values themeselves.
    :param scalarsInterpolated: The interpolated scalars, can either be a string or the values themselves.
    :param scalar: The name of the scalar being compared (e.g. "BVTV", "DA").
    :return: Returns a small dataframe with the mean for an individual before and after interpolation.
    """
    if isinstance(scalarsOriginal, str):
        scalarsOriginal = pd.read_csv(scalarsOriginal)
        if str(scalarsOriginal.columns[0]).lower() != "x":
            print("Scalar values likely from Paraview, using old setup...")
            scalarsOriginal = scalarsOriginal.iloc[:, 0:1]
        else:
            scalarsOriginal = scalarsOriginal.iloc[:, 3:4]
    if isinstance(scalarsOriginal, (pd.DataFrame, pd.Series, np.ndarray)):
        pass
    else:
        print("Original scalars input not understood...")

    if isinstance(scalarsInterpolated, str):
        scalarsInterpolated = pd.read_csv(scalarsInterpolated)
        if str(scalarsInterpolated.columns[0]).lower() != "x":
            print("Scalar values likely from Paraview, using old setup...")
            scalarsInterpolated = scalarsInterpolated.iloc[:, 0:1]
        else:
            scalarsInterpolated = scalarsInterpolated.iloc[:, 3:4]
    if isinstance(scalarsOriginal, (pd.DataFrame, pd.Series, np.ndarray)):
        pass
    else:
        print("Interpolated scalars input not understood...")

    original_mean = pd.DataFrame(scalarsOriginal.mean(axis=0))
    original_mean["scalar"] = f"{scalar}_Original"
    original_mean.set_index("scalar", inplace=True)
    original_mean.columns = [f"{name_in}"]
    interpolated_mean = pd.DataFrame(scalarsInterpolated.mean(axis=0))
    interpolated_mean["scalar"] = f"{scalar}_Interpolated"
    interpolated_mean.set_index("scalar", inplace=True)
    interpolated_mean.columns = [f"{name_in}"]
    joined = pd.concat([original_mean, interpolated_mean], axis=0)
    return joined


def map_multiple_scalars(
    base_name,
    registered_grid,
    original_xyz,
    canonical_point_cloud,
    canonical_geo,
    point_distance,
    scalar_list="",
    max_noarmalized=True,
    alternate_name="",
):
    """
    Funciton to map multiple scalars onto a deformed point cloud using SciPy nearest neighbor interpolation.
    Returns mapped point clouds and case files for each scalar in a list. Default is BVTV and DA.
    :param base_name:
    :param registered_grid:
    :param original_xyz:
    :param canonical_point_cloud:
    :param canonical_geo:
    :param point_distance:
    :param scalar_list: List of scalars to be mapped, e.g. ["BVTV", "DA"]. These values should be in the naming convention.
    :param max_normalized: If max normalized values be produced should.
    :param alternate_name: If the naming convention isn't consistent between point clouds and their originals.
    :return:
    """
    if scalar_list == "":
        scalar_list = ["BVTV", "DA"]

    for scalar in scalar_list:
        print(f"Mapping {scalar} for {base_name}...")
        if alternate_name == "":
            scalar_name = f"{base_name}_{scalar}_point_cloud_{point_distance}mm.csv"
        else:
            scalar_name = str(alternate_name)
        scalar_values = pd.read_csv(scalar_name, sep=",")
        if str(scalar_values.columns[0]).lower() != "x":
            print("Scalar values likely from Paraview, using old setup...")
            scalar_values = scalar_values.iloc[:, 0:1]
        else:
            scalar_values = scalar_values.iloc[:, 3:4]
        map_cloud(
            name_in=base_name,
            grid=registered_grid,
            points=original_xyz,
            original=canonical_point_cloud,
            canonical_geo=canonical_geo,
            scalar=str(scalar),
            scalar_values=scalar_values,
        )
        if max_noarmalized == True:
            map_cloud_max_normalized(
                name_in=base_name,
                grid=registered_grid,
                points=original_xyz,
                original=canonical_point_cloud,
                canonical_geo=canonical_geo,
                scalar=str(scalar),
                scalar_values=scalar_values,
            )


def find_using_oldname(oldname, scalar=False, suffix=False):
    """
    Convienence function for finding a short name using the oldname format.
    :param oldname: Medtool oldname or long name.
    :param scalar: Scalar value that is mismatched
    :return: Returns a list of matching string using glob
    """
    base_name = oldname
    oldname = oldname.split("_")

    if suffix == False:
        shortname = f"*{oldname[0]}{oldname[1]}*"
        if scalar != False:
            print(f"Appending {scalar} to search..")
            shortname = f"{shortname}_{scalar}_*"
    else:
        suffix = str(suffix)
        print(f"Appending {suffix} to search..")
        if scalar != False:
            print(f"Appending {scalar} to search..")
            suffix = f"_{scalar}_*{suffix}"
        shortname = f"*{oldname[0]}{oldname[1]}*{suffix}"
    print(f"Matching with {shortname}")
    shortname = glob.glob(str(shortname))

    if len(shortname) > 1:
        print(f"More than one name match for {base_name}!")
    elif len(shortname) == 0:
        print(f"No short match for {base_name}!")
    else:
        shortname = shortname[0]
        return shortname


# TODO Finish writing find_using_shortname
def find_using_shortname(shortname, bone, group_length=2, scalar=False, suffix=False):
    """
    Convienence funciton for finding a long name (Medtool oldname) using a shortname. WIP.
    :param shortname: Short name used to match with long name.
    :param bone: The bone that is being worked with, e.f. "Talus"
    :param group_length: The length of the group name as an integer (e.g. BE would be 2)
    :param scalar: If a scalar is being used in the naming convention add it here (e.g. "BVTV")
    :param suffix: A string to narrow down the choices.
    :return: Will return a single name match.
    """

    base_name = shortname
    group = base_name[:group_length]
    bone_start = base_name.find(str(bone))
    identifier = base_name[group_length:bone_start]
    print(f"Group set as {group}, identifier as {identifier}...")
    if suffix == False:
        long_name = f"{group}_{identifier}*"
        if scalar != False:
            print(f"Appending {scalar} to search..")
            long_name = f"{long_name}*_{scalar}_*"
    else:
        suffix = str(suffix)
        print(f"Appending {suffix} to search..")
        if scalar != False:
            print(f"Appending {scalar} to search..")
            suffix = f"*_{scalar}_*{suffix}"
        long_name = f"{group}_{identifier}*{suffix}"
    print(f"Matching with {shortname}")
    long_name = glob.glob(long_name)

    if len(long_name) > 1:
        print(f"More than one name match for {base_name}!")
    elif len(long_name) == 0:
        print(f"No short match for {base_name}!")
    else:
        long_name = long_name[0]
        return long_name


def find_vector(xyz1, xyz2):
    """
    Adjusted from https://stackoverflow.com/questions/51272288/how-to-calculate-the-vector-from-two-points-in-3d-with-python
    :param xyz1:
    :param xyz2:
    :return:
    """
    # setting unitSphere to True will make the vector scaled down to a sphere with a radius one, instead of it's orginal length
    final_vector = [xyz2[dimension] - coord for dimension, coord in enumerate(xyz1)]
    vector_length = np.linalg.norm(final_vector)
    return final_vector, vector_length


def df_vector(row):
    "Function to apply the find vectors row by row to a dataframe"
    xyz1 = row[0:3].values
    xyz2 = row[3:].values
    final_vector, magnitude = find_vector(xyz1, xyz2)
    return final_vector[0], final_vector[1], final_vector[2], magnitude


def align_vtks(
    rotation_dict, vtk_list, point_cloud_dir, auto3dgm_dir, substring_remove="Prox"
):
    directory = point_cloud_dir
    auto3d_dir = pathlib.Path(auto3dgm_dir)
    for key, value in rotation_dict.items():
        search_item = value + "_consolidated.vtk"
        if substring_remove in search_item:
            search_item = search_item.reaplce("Prox", "")
        found = [str(search_item) in vtk for vtk in vtk_list]
        found = [item for item, tru in enumerate(found) if tru]
        if len(found) == 1:
            print(vtk_list[found[0]])
        elif len(found) > 1:
            print(f"Found multiple match for {value}...")
        else:
            print(f"\n\n\nNo match found for {value}!\n\n\n")

        rotation_name = str(auto3d_dir.joinpath(str(key)))
        rotation = pd.read_csv(rotation_name)

        if np.linalg.det(rotation.iloc[:, :3].values) < 0:
            print(
                "\nDeterminant is negative, and the resulting points will be flipped...\n"
            )
        rotation["center"] = (0, 0, 0)
        data = pd.DataFrame({"x": [0], "y": [0], "z": [0], "center": [1]})
        rotation = rotation.append(data)

        mesh = pv.read(search_item)
        mesh.transform(rotation.values)
        mesh.points = mesh.points - mesh.center

        mesh.save(f"{value}_aligned.vtk")
        point_check = pd.DataFrame(mesh.points)
        point_check.columns = ["x", "y", "z"]
        point_check_save = directory.joinpath("rigid").joinpath(
            f"{value}_aligned_check.csv"
        )
        point_check.to_csv(str(point_check_save), index=False)


def get_initial_aligned_dict(auto3dgm_dir, csv_list=False):
    auto3d_dir = pathlib.Path(auto3dgm_dir)
    print(f"Getting matching dictionary from rotation matrices...\n")
    # Find the rotation matrix and then reverse name match for the initial rigid
    rotation_names = glob.glob(str(auto3d_dir.joinpath("*_rotation_matrix.txt")))
    rotation_names = [rotation.rsplit("\\", 1)[-1] for rotation in rotation_names]
    print(f"Found {len(rotation_names)} matrices in auto3dgm folder....\n")

    # This is ugly and I hate it, but it works.
    rotation_values = [
        "_".join(rotation.split("_", 2)[0:2])
        + rotation.split("_", 2)[-1]
        .replace("_rotation_matrix.txt", "")
        .replace("_", "")
        for rotation in rotation_names
    ]

    rotation_dict = dict(zip(rotation_names, rotation_values))
    if csv_list:
        vtk_list = glob.glob("*.csv")
    else:
        vtk_list = glob.glob("*.vtk")
    vtk_list.sort()
    print(f"Found {len(vtk_list)} vtk files....\n")

    if len(vtk_list) != len(rotation_names):
        print(
            "There number of rotation matrices is not equal to the number of meshes\nThis will end in tears..."
        )
    return rotation_dict, vtk_list


def get_rotation_dict(
    histomorph_vtk_dir: Union[str, pathlib.Path], csv_list: list = False
) -> dict:
    histomorph_vtk_dir = pathlib.Path(histomorph_vtk_dir)
    print(f"Getting matching dictionary from rotation matrices...\n")
    # Find the rotation matrix and then reverse name match for the initial rigid
    rotated_names = [item.name for item in histomorph_vtk_dir.glob("*.vtk") if "canonical" not in item.name]
    
    print(f"Found {len(rotated_names)} matrices in histomorph vtk folder....\n")

    # This is ugly and I hate it, but it works.
    # TODO this will be whatever we call the low res stuff
    rotation_values = []

    rotation_dict = dict(zip(rotated_names, rotation_values))

    return rotation_dict, rotated_names




def setup_point_cloud_folder_struct(point_cloud_folder):
    """
    Function to create the appropriate point cloud folder structure.
    """
    directory = pathlib.Path(point_cloud_folder)
    folder_list = ["rigid", "affine", "mapping", "deformable"]
    print("Checking if folder structure exists...")

    for folder in folder_list:
        check_folder = directory.joinpath(str(folder))
        if check_folder.exists() == False:
            print(f"Making {str(check_folder)}...\n")
            pathlib.Path.mkdir(check_folder)


# Use a for loop to process each point cloud file (i.e. f) in the list
def batch_initial_rigid(rotation_dict, auto3d_dir, point_cloud_dir, match="long_name"):
    auto3d_dir = pathlib.Path(auto3d_dir)
    directory = pathlib.Path(point_cloud_dir)
    for key, value in rotation_dict.items():
        if match == "long_name":
            base_name = str(key).replace("_rotation_matrix.txt", "")
        if match == "short_name":
            base_name = str(value)
        print(base_name)
        pc_match = glob.glob(f"*{base_name}*")
        if len(pc_match) == 0:
            substring_check = "trabecular"
            if substring_check in base_name:
                print(
                    "Found 'trabecular' in the naming scheme. Make certain you want to align to a trab auto3dgm instead of cortical!"
                )
                base_name = base_name.replace("trabecular", "")
                pc_match = glob.glob(f"*{base_name}*")
        if len(pc_match) == 0:
            print(f"No point cloud match for {key} found!)")
        elif len(pc_match) > 1:
            print(f"More than one point cloud match for {key} found!)")
        else:
            print(f"Input pointcloud name {pc_match}.")
            print("Initial registration with", base_name)
            moving_pc = pd.read_csv(pc_match[0], sep=",")
            moving_pc = moving_pc.iloc[:, 1:].values
            print(moving_pc.shape)
            try:
                initial_rigid(
                    name_in=base_name,
                    moving=moving_pc,
                    auto3d_dir=auto3d_dir,
                    auto3d_ref="",
                    rotation_matrix="",
                    origin="",
                    outdir=directory.joinpath("rigid"),
                )
            except FileNotFoundError:
                initial_rigid(
                    name_in=base_name[:-1],
                    moving=moving_pc,
                    auto3d_dir=auto3d_dir,
                    auto3d_ref="",
                    rotation_matrix="",
                    origin="",
                    outdir=directory.joinpath("rigid"),
                )
        print("\n")


def batch_set_origin_zero(point_cloud_folder):

    rigid_folder = pathlib.Path(point_cloud_folder).joinpath("rigid")

    # Set the origin of the aligned point clouds to 0 so they are all centered:
    aligned_fileList = glob.glob(str(rigid_folder.joinpath("*_aligned_original.csv")))
    print(f"Setting the {len(aligned_fileList)} point clouds origin to 0...")

    # Use a for loop to process each point cloud file (i.e. f) in the list
    for files in aligned_fileList:
        name_In = files.replace("_aligned_original.csv", "")
        name_In = name_In.split("\\")[-1]
        name_In = name_In.split("/")[-1]
        print("Initial registration with", name_In)
        moving = pd.read_csv(files, sep=",", header=None).values
        print(moving.shape)
        set_origin_to_zero(name_In, moving, outDir=rigid_folder)
        print("\n")
    print("\nDone!")


def batch_register_deformed_and_high_res(rotation_dict, point_cloud_dir):
    """
    This function only exists because I deleted something I shouldn't have.
    :param rotation_dict:
    :param point_cloud_dir:
    :return:
    """

    rigid_directory = point_cloud_dir.joinpath("rigid")
    deformed_directory = point_cloud_dir.joinpath("deformable")
    current_len = len(rotation_dict)
    for keys, values in rotation_dict.items():
        print(keys)
        highres = f"{values}_aligned_check.csv"
        fixed = pd.read_csv(rigid_directory.joinpath(highres), sep=",")
        moving_name = keys.replace("_rotation_matrix.txt", "")
        if moving_name[-1] == "_":
            print(
                "\nSome of you aren't bothered by double underscores and it shows...\n"
            )
            moving_name = moving_name[:-1]
        moving_name = f"{moving_name}_deformable_moving"
        print(f"Registering {moving_name}...")
        moving = glob.glob(str(deformed_directory.joinpath(f"*{moving_name}*")))
        moving = pd.read_csv(moving[0], sep=",", header=None)

        rigid_low_to_high(
            name_in=moving_name,
            moving=moving.values,
            fixed=fixed.values,
            iterations=50,
            tolerance=0.001,
            matrix=False,
            outDir=point_cloud_dir,
        )
        current_len -= 1
        print(f"\n{current_len} point clouds left to register...\n")


def batch_register_low_and_high_res(rotation_dict, point_cloud_dir):

    rigid_directory = point_cloud_dir.joinpath("rigid")
    current_len = len(rotation_dict)
    for keys, values in rotation_dict.items():
        print(keys)
        highres = f"{values}_aligned_check.csv"
        fixed = pd.read_csv(rigid_directory.joinpath(highres), sep=",")
        moving_name = keys.replace("_rotation_matrix.txt", "")
        if moving_name[-1] == "_":
            print(
                "\nSome of you aren't bothered by double underscores and it shows...\n"
            )
            moving_name = moving_name[:-1]
        moving = glob.glob(str(rigid_directory.joinpath(f"*{moving_name}*")))
        if not moving:
            moving = glob.glob(
                str(
                    rigid_directory.joinpath(
                        f"*{moving_name.replace('_trabecular', '')}*"
                    )
                )
            )
        moving = pd.read_csv(moving[0], sep=",", header=None)
        if type(moving.loc[0][0]) == str:
            moving = moving.iloc[1:, :].astype(float)

        rigid_low_to_high(
            name_in=moving_name,
            moving=moving.values,
            fixed=fixed.values,
            iterations=50,
            tolerance=0.001,
            matrix=False,
            outDir=point_cloud_dir,
        )
        current_len -= 1
        print(f"\n{current_len} point clouds left to register...\n")


def batch_rigid(point_cloud_dir, canonical, iterations=100, tolerance=0.001):

    rigid_directory = pathlib.Path(point_cloud_dir).joinpath("rigid")

    if type(canonical) == list:
        canonical_pc = canonical[0]
    else:
        canonical_pc = canonical

    # Use glob to match text of a file name and build a list of the point clouds to be aligned
    rigid_fileList = glob.glob(str(rigid_directory.joinpath("*_aligned_original.csv")))

    # Sort the file list by ascending order
    rigid_fileList.sort()

    # Print to console for verification
    print(rigid_fileList)

    # Print the length of the list to verify it is finding all the files
    current_len = len(rigid_fileList)

    #moving = pd.read_csv(canonical_pc, sep=",", header=None)
    #if type(moving.iloc[0][0]) in [str, object]:
    moving = pd.read_csv(canonical_pc, sep=",")    
    print(f"There are {len(moving)} points in the moving cloud")

    # Loop over the list and perform the rigid registration
    for file in rigid_fileList:
        name_In = file.replace(".csv", "")
        if platform.system().lower() == "windows":
            name_In = name_In.split("\\")[-1]
        else:
            name_In = name_In.split("/")[-1]

        print(f"Rigid registration with {name_In}")

        #fixed = pd.read_csv(str(file), header=None, sep=",")
        fixed = pd.read_csv(str(file), sep=",")
        print_fixed_points(fixed)
        print(f"There are {len(fixed)} points in the fixed cloud")
        print("\n")
        # Perform the rigid registration with 100 iterations or an error of 0.05
        rigid(
            name_in=name_In,
            moving=moving.values,
            fixed=fixed.values,
            iterations=int(iterations),
            tolerance=float(tolerance),
            outDir=point_cloud_dir,
        )
        print("\n")
        current_len -= 1
        print(f"{current_len} point clouds left to register...\n")


def batch_affine(point_cloud_dir, iterations=100, tolerance=0.001):
    rigid_directory = pathlib.Path(point_cloud_dir).joinpath("rigid")
    affine_fileList = glob.glob(str(rigid_directory.joinpath("*_rigid_moving.csv")))
    affine_fileList.sort()
    print(affine_fileList)
    current_len = len(affine_fileList)

    for file in affine_fileList:
        moving = pd.read_csv(file, header=None, sep=",")
        if moving.iloc[0][0] == 'x':
            moving = pd.read_csv(file, sep=",")

        moving = moving.values
        print(f"There are {len(moving)} points in the moving cloud")
        print_moving_points(moving)

        name_In = file.replace("_aligned_original_rigid_moving.csv", "")
        if platform.system().lower() == "windows":
            name_In = name_In.split("\\")[-1]
        else:
            name_In = name_In.split("/")[-1]

        print(f"Performing affine registration with {name_In}...")

        fixed_file = file.replace("_rigid_moving.csv", ".csv")          
        fixed = pd.read_csv(fixed_file, header=None, sep=",")
        if fixed.iloc[0][0] == 'x':
            fixed = pd.read_csv(fixed_file, sep=",").values
        print(f"There are {len(fixed)} points in the fixed cloud")
        print_fixed_points(fixed)
        print("\n")
        affine(
            name_in=name_In,
            moving=moving,
            fixed=fixed,
            iterations=int(iterations),
            tolerance=float(tolerance),
            outDir=point_cloud_dir,
        )
        print("\n")
        current_len -= 1
        print(f"{current_len} point clouds left to register...\n")


def batch_deformable(point_cloud_dir, iterations=100, tolerance=0.001):
    rigid_directory = pathlib.Path(point_cloud_dir).joinpath("rigid")
    affine_directory = pathlib.Path(point_cloud_dir).joinpath("affine")
    deformble_fileList = glob.glob(
        str(affine_directory.joinpath("*_affine_moving.csv"))
    )
    deformble_fileList.sort()
    print(deformble_fileList)

    current_len = len(deformble_fileList)

    for file in deformble_fileList:
        moving = pd.read_csv(file, header=None, sep=",")
        if moving.iloc[0][0] == 'x':
            moving = pd.read_csv(file, sep=",")
        moving = moving.values

        print(f"There are {len(moving)} points in the moving cloud")
        print_moving_points(moving)

        name_In = file.replace("_affine_moving.csv", "")
        if platform.system().lower() == "windows":
            name_In = name_In.split("\\")[-1]
        else:
            name_In = name_In.split("/")[-1]

        print(f"Performing deformable registration with {name_In}...")

        fixed_file = rigid_directory.joinpath(f"{name_In}_aligned_original.csv")
        fixed = pd.read_csv(fixed_file, header=None, sep=",")
        if fixed.iloc[0][0] == 'x':
            fixed = pd.read_csv(fixed_file, sep=",")
        fixed = fixed.values
        print(f"There are {len(fixed)} points in the fixed cloud")
        print_fixed_points(fixed)
        print("\n")
        deformable(
            name_in=name_In,
            moving=moving,
            fixed=fixed,
            iterations=int(iterations),
            tolerance=float(tolerance),
            outDir=point_cloud_dir,
        )
        print("\n")
        current_len -= 1
        print(f"{current_len} point clouds left to register...\n")


def map_cloud_from_vtk(
    name_in,
    scalar_vtk,
    deformed_points,
    point_cloud_dir,
    canonical_pc: pd.DataFrame,
    canonical_vtk,
):

    start = timer()

    # Set up the directory and names for the files output
    mapped_dir = point_cloud_dir.joinpath("mapping")
    deformed_cloud_name = mapped_dir.joinpath(f"{name_in}_deformed_mapped.csv")
    original_cloud_name = mapped_dir.joinpath(f"{name_in}_original_mapped.csv")
    vtk_name = mapped_dir.joinpath(f"{name_in}_original_mapped.vtk")

    # Get the registered points to map them to
    mapped_cloud = deformed_points
    mapped_cloud.columns = ["x", "y", "z"]
    canonical_pc.columns = ["x", "y", "z"]

    to_map = mapped_cloud.values

    # Read in the mesh file containing all the scalars
    mesh = pv.read(scalar_vtk)

    # Convert cell data to point data and then get the xyz
    mesh = mesh.cell_data_to_point_data()
    points = pd.DataFrame(mesh.points)

    # Read in the canonical so the points can be pasted over
    canonical_vtk = pv.read(canonical_vtk)

    # Loop through the point array scalars that are present and map them onto the cloud
    for key, values in mesh.point_arrays.items():
        scalar = str(key)
        scalar_values = values
        mapped = pd.DataFrame(
            griddata(
                points=points.values, values=scalar_values, xi=to_map, method="nearest"
            ),
            columns=[scalar],
        )
        mapped_cloud = pd.concat([mapped_cloud, mapped], axis=1)

    # Get the mapped scalars without the deformed xyz and paste them onto the undeformed canonical point cloud
    original_cloud_scalars = mapped_cloud.iloc[:, 3:].astype(float)

    original_cloud = pd.concat([canonical_pc, original_cloud_scalars], axis=1)

    # Take the scalars and paste them onto the canonical vtk
    for column in original_cloud_scalars:
        print(f"Mapping {column}...\n")
        canonical_vtk[str(column)] = original_cloud_scalars[str(column)]

    # Save the mapped values
    mapped_cloud.to_csv(str(deformed_cloud_name), sep=",", index=False)
    original_cloud.to_csv(str(original_cloud_name), sep=",", index=False)
    canonical_vtk.save(str(vtk_name))

    print()
    _end_timer(start_timer=start, message="Mapping")


def batch_mapping(
    rotation_dict: dict,
    point_cloud_dir: str,
    canonical_pc: str,
    canonical_vtk=False,
    canonical_geo=False,
):
    directory = point_cloud_dir
    deformable_dir = directory.joinpath("deformable")
    for key, value in rotation_dict.items():
        registered_cloud = deformable_dir.joinpath(
            key.replace("_rotation_matrix.txt", "_deformable_moving.csv")
        )
        if not registered_cloud.exists():
                registered_cloud = deformable_dir.joinpath(f"{value}_deformable_moving.csv")
        registered_cloud = pd.read_csv(registered_cloud, header=None)        
        if registered_cloud.iloc[0][0] == "x":
            registered_cloud = pd.read_csv(registered_cloud)        
        vtk_mesh = f"{value}_aligned.vtk"
        name_in = str(value)
        if canonical_geo:
            canonical_vtk = canonical_geo[0].replace(".geo", ".vtk")
        if canonical_vtk:
            canonical_vtk = canonical_vtk
        if type(canonical_pc) == list:
            canonical_points = pd.read_csv(canonical_pc[0], header=None)            
        else:
            canonical_points = pd.read_csv(canonical_pc, header=None)
        if canonical_points.iloc[0][0] == "x":
            if type(canonical_pc) == list:
                canonical_points = pd.read_csv(canonical_pc[0])
            else:
                canonical_points = pd.read_csv(canonical_pc, header=None)        
        if len(canonical_points) - len(registered_cloud) == 1:
            registered_cloud.loc[-1] = registered_cloud.loc[0]
        map_cloud_from_vtk(
            name_in=name_in,
            scalar_vtk=vtk_mesh,
            deformed_points=registered_cloud,
            point_cloud_dir=directory,
            canonical_pc=canonical_points,
            canonical_vtk=canonical_vtk,
        )


def gather_scalars(point_cloud_dir, canonical_vtk, max_normalized=True):
    start = timer()

    mapped_dir = point_cloud_dir.joinpath("mapping")
    results_dir = point_cloud_dir.joinpath("results")
    mean_mesh_outname = results_dir.joinpath(canonical_vtk.replace(".vtk", "_mean.vtk"))
    mean_mesh_max_outname = results_dir.joinpath(
        canonical_vtk.replace(".vtk", "_mean_max_normalized.vtk")
    )

    if not results_dir.exists():
        print(f"Making {str(results_dir)}...\n")
        pathlib.Path.mkdir(results_dir)

    mapped_list = glob.glob(str(mapped_dir.joinpath(f"*_original_mapped.csv")))

    # removed this, but am waiting to see if it throws errors with a newer data
    # column_list = []

    scalar_means = pd.DataFrame()
    for mapped in mapped_list:
        base_name = pathlib.Path(mapped).name.replace("_original_mapped.csv", "")
        scalar_values = pd.read_csv(mapped)
        scalar_values = scalar_values.iloc[:, 3:]
        means = pd.DataFrame(scalar_values.mean(axis=0))
        means.columns = ["mean"]
        # column_list.extend([str(base_name)])
        print(f"\n\n{base_name} means:\n{means}\n")
        scalar_means = pd.concat([scalar_means, means.T], axis=1)

    # scalar_means.columns = column_list
    scalar_means.to_csv(str(results_dir.joinpath("mapped_scalar_means.csv")), sep=",")

    # Scalars are expected to be at the end of the name, and we can get the unique versions by using a set
    scalar_list = list(scalar_means.columns)
    scalar_list = [item.rpartition("_")[-1] for item in scalar_list]
    scalar_list = list(set(scalar_list))

    if not max_normalized:
        mesh = map_canonical_vtk(
            scalar_list=scalar_list,
            mapped_list=mapped_list,
            canonical_vtk=canonical_vtk,
            outDir=results_dir,
            max_normailzed=False,
        )
        mesh.save(str(mean_mesh_outname))
    else:
        mesh, max_mesh = map_canonical_vtk(
            scalar_list=scalar_list,
            mapped_list=mapped_list,
            canonical_vtk=canonical_vtk,
            outDir=results_dir,
            max_normailzed=True,
        )

        mesh.save(str(mean_mesh_outname))
        max_mesh.save(str(mean_mesh_max_outname))
    _end_timer(start_timer=start, message="Generating results")


def map_canonical_vtk(
    scalar_list,
    mapped_list,
    canonical_vtk,
    outDir,
    group_name="",
    max_normailzed=True,
    write_results=True,
):
    mesh = pv.read(canonical_vtk)
    if max_normailzed == True:
        max_mesh = mesh.copy()

    df_list = [pd.read_csv(file) for file in mapped_list]
    df_list = [df.iloc[:, 3:] for df in df_list]
    df_list = pd.concat(df_list, axis=1)
    max_norm = lambda col: col / col.max()

    base_name = [mapped.replace("_original_mapped.csv", "") for mapped in mapped_list]
    column_list = [str(pathlib.Path(name).parts[-1]) for name in base_name]

    for scalar in scalar_list:
        print(f"Mapping {scalar} to vtk...")
        scalar_name = scalar
        if group_name != "":
            scalar_name = f"{group_name}_{scalar_name}"
        empty_scalar = pd.DataFrame()
        current_scalars = [item for item in df_list.columns if f"_{scalar}" in item]
        empty_scalar = [
            pd.concat([empty_scalar, df_list[df]], axis=1) for df in current_scalars
        ]
        scalar_df = pd.concat(empty_scalar, axis=1)
        scalar_df.columns = column_list
        scalar_mean = scalar_df.mean(axis=1)
        standard_dev = scalar_df.std(axis=1)
        coef_var = scalar_df.std(axis=1) / scalar_df.mean(axis=1)

        if write_results == True:
            scalar_outname = outDir.joinpath(f"{scalar}_results.csv")
            scalar_df.to_csv(scalar_outname, sep=",")

        mesh[str(scalar_name)] = scalar_mean
        mesh[f"{str(scalar_name)}_std"] = standard_dev
        mesh[f"{str(scalar_name)}_coef"] = coef_var

        if max_normailzed == True:
            scalar_df_max_norm = scalar_df.apply(max_norm, axis=0)

            if write_results == True:
                scalar_outname_max = str(scalar_outname).replace(
                    ".csv", "_max_normalized.csv"
                )
                scalar_df_max_norm.to_csv(scalar_outname_max, sep=",")

            scalar_mean_max = scalar_df_max_norm.mean(axis=1)
            standard_dev_max = scalar_df_max_norm.std(axis=1)
            coef_var_max = scalar_df_max_norm.std(axis=1) / scalar_df_max_norm.mean(
                axis=1
            )

            max_mesh[f"{str(scalar_name)}_max_norm"] = scalar_mean_max
            max_mesh[f"{str(scalar_name)}_max_norm_std"] = standard_dev_max
            max_mesh[f"{str(scalar_name)}_max_norm_coef"] = coef_var_max

    if max_normailzed == True:
        return mesh, max_mesh
    else:
        return mesh


def make_VTP_point_cloud(xyz_points, scalar="", scalar_name=""):
    """
    WIP
    Adapted from https://github.com/grosenkj/telluricpy/blob/master/telluricpy/dataFiles/XYZtools.py
    :param locXYZ:
    :return:
    """
    # Ignore a deprecation warning we can't do anything about
    import warnings

    warnings.filterwarnings("ignore")

    # Load the file
    if isinstance(xyz_points, np.ndarray):
        loc = xyz_points
    elif isinstance(xyz_points, str):
        loc = np.genfromtxt(xyz_points)

    # Make the pts
    vtkPts = vtk.vtkPoints()
    vtkPts.SetData(npsup.numpy_to_vtk(loc, deep=1))

    # Make the poly data
    polyPtsVTP = vtk.vtkPolyData()
    polyPtsVTP.SetPoints(vtkPts)

    if scalar != "":
        scalar = npsup.numpy_to_vtk(scalar, deep=1)
        scalar.SetName(scalar_name)
        polyPtsVTP.GetPointData().SetScalars(scalar)
    # Return
    return polyPtsVTP


def vtk_writeVTP(inputMesh, outName, outDir="", outType="ascii"):
    """
    WIP
    :param inputMesh:
    :param outName:
    :param outDir:
    :param outType:
    :return:
    """
    outName = str(outName) + ".vtp"
    outDir = _get_outDir(outDir)
    outFile = pathlib.Path(outDir).joinpath(outName)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetInputData(inputMesh)
    writer.SetFileName(str(outFile))
    if outType != "ascii":
        writer.SetDataModeToBinary()
    else:
        writer.SetDataModeToAscii()
    writer.Write()


def vtk_visualize_deformation(originalPoints, deformedPoints, magnitudeScalar):
    """
    WIP
    :param originalPoints:
    :param deformedPoints:
    :param magnitudeScalar:
    :return:
    """
    if isinstance(originalPoints, (pd.DataFrame, pd.Series)):
        points0 = originalPoints.values
    if isinstance(deformedPoints, (pd.DataFrame, pd.Series)):
        points1 = deformedPoints.values
    size = len(originalPoints)
    stacked = pd.concat([pd.DataFrame(points0), pd.DataFrame(points1)])

    pts = vtk.vtkPoints()
    pts.SetData(npsup.numpy_to_vtk(stacked.values, deep=1))

    lines = vtk.vtkCellArray()

    for i in range(size - 1):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i)  # the i is the index of the Origin point
        line.GetPointIds().SetId(
            1, i + size
        )  # i + size is the index of second point array
        lines.InsertNextCell(line)

    linesPolyData = vtk.vtkPolyData()
    linesPolyData.SetPoints(pts)
    linesPolyData.SetLines(lines)
    linesPolyData.GetCellData().SetScalars(npsup.numpy_to_vtk(magnitudeScalar, deep=1))
    return linesPolyData


def visualize_registration_movement(point_cloud_dir, canonical_pc):
    # Visualize the amount of movement in the point clouds

    rigid_dir = pathlib.Path(point_cloud_dir).joinpath("rigid")
    deformable_dir = pathlib.Path(point_cloud_dir).joinpath("deformable")
    diagnostics_dir = pathlib.Path(point_cloud_dir).joinpath("diagnostics")

    if pathlib.Path(diagnostics_dir).exists() == False:
        print(f"Creating diagnostics folder {diagnostics_dir}")
        pathlib.Path.mkdir(diagnostics_dir)

    canonical_points = pd.read_csv(canonical_pc, header=None)

    rigid_fileList = glob.glob(
        str(rigid_dir.joinpath("*_aligned_original_rigid_moving.csv"))
    )
    rigid_fileList.sort()
    print(len(rigid_fileList))

    # Create and empty dataframe to append distance values to
    all_distances = pd.DataFrame()
    x_vectors = pd.DataFrame()
    y_vectors = pd.DataFrame()
    z_vectors = pd.DataFrame()
    magnitudes = pd.DataFrame()

    # Loop through the rigid file list and load the corresponding deformable point cloud
    # Then calculate the distance between the rows for each point and write out a new point cloud
    for f in rigid_fileList:
        name_In = f.replace("_aligned_original_rigid_moving.csv", "")
        if platform.system().lower() == "windows":
            name_In = name_In.split("\\")[-1]
        else:
            name_In = name_In.split("/")[-1]
        print(name_In)
        rigid_df = pd.read_csv(f, header=None)
        deformable = pd.read_csv(
            str(deformable_dir.joinpath(f"{name_In}_deformable_moving.csv")),
            header=None,
        )
        distance = pd.DataFrame(get_distance(rigid_df, deformable))
        all_distances = pd.concat([all_distances, distance], axis=1)
        rigid_df = pd.concat([rigid_df, distance], axis=1)
        rigid_df.columns = ["x", "y", "z", "distance"]
        print(rigid_df)
        rigid_df.to_csv(
            str(diagnostics_dir.joinpath(f"{name_In}_registration_movement.csv")),
            sep=",",
        )
        new_df = pd.concat([pd.DataFrame(rigid_df.iloc[:, :3]), deformable], axis=1)
        new_df.columns = ["x1", "y1", "z1", "x2", "y2", "z3"]
        temp_df = new_df.apply(df_vector, axis=1)
        temp_df = pd.DataFrame(temp_df, columns=["temp"])
        vectors = pd.DataFrame(
            temp_df["temp"].tolist(),
            index=temp_df.index,
            columns=["xdif", "ydif", "zdif", "magnitude"],
        )
        x_vectors = pd.concat([x_vectors, vectors["xdif"]], axis=1)
        y_vectors = pd.concat([y_vectors, vectors["zdif"]], axis=1)
        z_vectors = pd.concat([z_vectors, vectors["zdif"]], axis=1)
        magnitudes = pd.concat([magnitudes, vectors["magnitude"]], axis=1)

    # Calculate the mean distance for each point of the point cloud across the sample
    mean_distance = all_distances.mean(axis=1)
    x_vectors = x_vectors.mean(axis=1)
    y_vectors = y_vectors.mean(axis=1)
    z_vectors = z_vectors.mean(axis=1)
    magnitudes = magnitudes.mean(axis=1)

    mean_vectors = pd.concat([x_vectors, y_vectors, z_vectors, magnitudes], axis=1)
    mean_vectors.columns = ["xdif", "ydif", "zdif", "magnitude"]

    canonical_points["Mean_deformation"] = mean_distance
    canonical_points.columns = ["x", "y", "z", "Mean_deformation"]
    canonical_points.to_csv(
        str(diagnostics_dir.joinpath("Mean_deformation.csv")), index=False
    )

    end_points = pd.DataFrame(
        canonical_points.iloc[:, :3].values + mean_vectors.iloc[:, :3].values
    )

    difference_lines = vtk_visualize_deformation(
        originalPoints=canonical_points.iloc[:, :3],
        deformedPoints=end_points,
        magnitudeScalar=magnitudes,
    )
    vtk_writeVTP(
        inputMesh=difference_lines,
        outName="Mean_deformation",
        outDir=diagnostics_dir,
        outType="binary",
    )


def get_mean_vtk_groups(
    group_list: list,
    group_identifiers: list,
    bone: str,
    canonical_vtk: Union[str, pathlib.Path],
    point_cloud_dir: Union[str, pathlib.Path],
    max_normalized: bool = True,
):

    mapping_dir = pathlib.Path(point_cloud_dir).joinpath("mapping")
    results_dir = pathlib.Path(point_cloud_dir).joinpath("results")
    if not results_dir.exists():
        pathlib.Path.mkdir(results_dir)
    dictionary = dict(zip(group_list, group_identifiers))
    print(dictionary)
    for key, values in dictionary.items():
        if type(values) == str:
            identifier = values
            try:
                group_scalars = glob.glob(
                    str(mapping_dir.joinpath(f"*{identifier}*_original_mapped.csv"))
                )
                group_scalars.sort()
            except:
                print(f"No mapped files for {key} found....\n")
        elif type(values) == list:
            identifier_list = values
            group_scalars = []
            for identifier in identifier_list:
                try:
                    group_scalars.extend(
                        glob.glob(
                            str(
                                mapping_dir.joinpath(
                                    f"*{identifier}*_original_mapped.csv"
                                )
                            )
                        )
                    )
                except:
                    print(
                        f"No mapped files matching {identifier} for {key} found....\n"
                    )
            group_scalars.sort()
            if len(group_scalars) != len(identifier_list):
                print(
                    f"\n\n\nFound {len(group_scalars)} individuals in {key} but expected {len(identifier_list)}:\n"
                )
                [print(group) for group in group_scalars]
        else:
            print("Group identifiers not understood....")

        if len(group_scalars) > 1:
            print(f"\n{key} list contains {len(group_scalars)} individuals...")
            outname = results_dir.joinpath(f"Mean_{key}_{bone}")
            scalar_list = pd.read_csv(group_scalars[0]).iloc[:, 3:]
            # Scalars are expected to be at name, following an
            scalar_list = [
                item.rpartition("_")[-1] for item in list(scalar_list.columns)
            ]
            scalar_list = list(set(scalar_list))
            if max_normalized == True:
                mesh, max_mesh = map_canonical_vtk(
                    scalar_list=scalar_list,
                    mapped_list=group_scalars,
                    canonical_vtk=canonical_vtk,
                    outDir=results_dir,
                    group_name=key,
                    max_normailzed=True,
                    write_results=False,
                )
                mesh.save(f"{outname}.vtk")
                max_mesh.save(f"{outname}_max_normalized.vtk")
            else:
                mesh = map_canonical_vtk(
                    scalar_list=scalar_list,
                    mapped_list=group_scalars,
                    canonical_vtk=canonical_vtk,
                    outDir=results_dir,
                    group_name=key,
                    max_normailzed=False,
                    write_results=False,
                )
                mesh.save(f"{outname}.vtk")
        else:
            print(f"Not enough individuals in {key} to create a mean.\n")


def vtp_visualize_deformation(point_cloud_dir, canonical_pc, outName):
    deformable_dir = pathlib.Path(point_cloud_dir).joinpath("deformable")
    diagnostic_dir = pathlib.Path(point_cloud_dir).joinpath("diagnostics")
    if outName == "":
        deformed_name = f"mean_deformed"
    else:
        deformed_name = f"mean_deformed_{outName}"

    deformable_list = glob.glob(str(deformable_dir.joinpath("*_deformable_moving.csv")))
    all_distances = pd.DataFrame()
    for deform in deformable_list:
        temp_df = pd.read_csv(deform, header=None)
        all_distances = pd.concat([all_distances, temp_df], axis=1)
    magnitudes_list = glob.glob(
        str(diagnostic_dir.joinpath("*_registration_movement.csv"))
    )
    magnitudes = [pd.read_csv(magnitude) for magnitude in magnitudes_list]
    temp_magnitude = pd.DataFrame()
    magnitudes = [
        pd.concat([temp_magnitude, frame.iloc[:, 4:]]) for frame in magnitudes
    ]
    magnitudes = pd.concat(magnitudes, axis=1)
    magnitudes = magnitudes.mean(axis=1)

    all_distances["mean_x"] = all_distances[0].mean(axis=1)
    all_distances["mean_y"] = all_distances[1].mean(axis=1)
    all_distances["mean_z"] = all_distances[2].mean(axis=1)

    mean_deformed = all_distances[["mean_x", "mean_y", "mean_z"]]
    mean_deformed.to_csv()
    if type(canonical_pc) == list:
        canonical_points = canonical_pc[0]
    else:
        canonical_points = canonical_pc

    mean_rigid = pd.read_csv(str(canonical_points), header=None)

    difference_lines = vtk_visualize_deformation(
        originalPoints=mean_rigid,
        deformedPoints=mean_deformed,
        magnitudeScalar=magnitudes,
    )
    vtk_writeVTP(
        inputMesh=difference_lines,
        outName=str(deformed_name),
        outDir=diagnostic_dir,
        outType="binary",
    )


def case_to_vtk(inputMesh, outName):
    """
    Function to convert a case and its scalars to a vtk file.
    :param inputMesh:
    :return:
    """
    mesh = vtk_read_case(inputMesh)
    conversion = pv.wrap(mesh)
    vtk_mesh = conversion.pop(0)
    outName.replace(".vtk", "")
    vtk_mesh.save(f"{outName}.vtk")


def vtk_read_case(inputMesh):
    """
    Function to read in engishtgold case files.
    :param inputMesh:
    :return:
    """
    start = timer()
    reader = vtk.vtkEnSightGoldReader()
    reader.SetCaseFileName(str(inputMesh))
    reader.Update()
    vtk_mesh = reader.GetOutput()
    print("\n")
    _vtk_print_mesh_info(vtk_mesh)
    print("\n")
    _end_timer(start, message="Reading in Case file")
    return vtk_mesh


def get_distplot(
    registered_dataset,
    scalar,
    fontsize=2,
    legendfont=40,
    xlim=[0.0, 1.0],
    colors="Paired",
    background="white",
):
    registered_dataset = pd.DataFrame(registered_dataset.groupby("group").mean()).T
    registered_dataset["Mean"] = registered_dataset.mean(axis=1)
    group_list = list(registered_dataset.columns)
    if isinstance(colors, list):
        colors = [str(x).lower() for x in colors]
        colors = sns.color_palette(colors)
    else:
        colors = sns.color_palette(str(colors), len(group_list))
    sns.set(rc={"figure.figsize": (24.4, 14.4)})
    sns.set(font_scale=fontsize)
    if background == "white":
        sns.set_style("white")
    sns.despine(left=True)

    fig, ax = plt.subplots()
    for r in range(len(group_list)):
        sns.distplot(
            registered_dataset[[str(group_list[r])]],
            hist=False,
            rug=True,
            label=str(group_list[r]),
            kde_kws={"lw": 5, "alpha": 1, "color": colors[r]},
            rug_kws={"alpha": 0.1, "color": colors[r]},
            ax=ax,
        ).set(xlim=(float(xlim[0]), float(xlim[1])))
    try:
        plt.setp(ax.get_legend().get_texts(), fontsize=legendfont)
    except AttributeError:
        plt.setp(ax.legend(fontsize=legendfont))
    ax.set(xlabel=str(scalar), ylabel="Counts")
    return fig


def consolidate_case(
    inputMesh,
    outName,
    nameMatch,
    scalars=["BVTV", "DA"],
    outputs=[
        "_original_average.case",
        "_original_coef_var.case",
        "_original_standard_dev.case",
    ],
    max_normazlied=True,
    pairwise=False,
):
    mesh = vtk_read_case(inputMesh)
    conversion = pv.wrap(mesh)
    vtk_mesh = conversion.pop(0)
    vtk_mesh.clear_arrays()
    nameMatch = list(nameMatch)
    if pairwise:
        possibilities = list(itertools.combinations(nameMatch, 2))
        consolidate_scalars = [
            f"{scalar}*{name[0]}_vs_{name[1]}"
            for name in possibilities
            for scalar in scalars
        ]
    else:
        consolidate_scalars = [
            f"{name}*{scalar}" for name in nameMatch for scalar in scalars
        ]
    for consolidate in consolidate_scalars:
        print(consolidate)
        for output in outputs:
            new_case = glob.glob(f"*{consolidate}*{output}*.case")
            # new_case = glob.glob(f"*{output}")
            print(new_case)
            if len(new_case) == 0:
                print("No case files match the consolidation criteria")
            elif len(new_case) == 1:
                temp_case = vtk_read_case(new_case[0])
                temp_conversion = pv.wrap(temp_case)
                temp_mesh = temp_conversion.pop(0)
                temp_array_name = list(temp_mesh.point_arrays.items())[0][0]
                temp_array_name = temp_array_name.replace("ESca1", "")
                temp_array_name = temp_array_name.replace("_original_", "_")
                temp_array = list(temp_mesh.point_arrays.items())[0][1]
                vtk_mesh[str(temp_array_name)] = temp_array
            else:
                for temp in new_case:
                    temp_case = vtk_read_case(temp)
                    temp_conversion = pv.wrap(temp_case)
                    temp_mesh = temp_conversion.pop(0)
                    temp_array_name = list(temp_mesh.point_arrays.items())[0][0]
                    temp_array_name = temp_array_name.replace("ESca1", "")
                    temp_array_name = temp_array_name.replace("_original_", "_")
                    temp_array_name = temp_array_name.replace("__", "_")
                    temp_array = list(temp_mesh.point_arrays.items())[0][1]
                    vtk_mesh[str(temp_array_name)] = temp_array
    outName = outName.replace(".vtk", "")
    vtk_mesh.save(f"{outName}.vtk")


def consolidate_vtk(
    input_mesh: str,
    out_name: str,
    name_match: Union[str, None],
    bayes_match: bool = False,
    scalars=["BVTV", "DA", "BSBV", "Tb_Sp", "Tb_Th"],
    pairwise=False,
):
    vtk_mesh = pv.read(input_mesh)
    vtk_mesh.clear_arrays()
    if bayes_match:
        consolidate_scalars = []
        for scalar in scalars:
            consolidate_scalars.append(glob.glob(f"*{scalar}*{name_match}*.vtk"))
        consolidate_scalars = list(itertools.chain.from_iterable(consolidate_scalars))
    elif pairwise:
        possibilities = list(itertools.combinations(name_match, 2))
        consolidate_scalars = [
            f"{scalar}*{name[0]}_vs_{name[1]}"
            for name in possibilities
            for scalar in scalars
        ]
    else:
        consolidate_scalars = [f"{name_match}{scalar}" for scalar in scalars]

    for consolidate in consolidate_scalars:
        print(f"\n Consolidating {consolidate}")
        #Had to add the .vtk because Nick is an ASSSSS
        temp_mesh = pv.read(f"{consolidate}.vtk")
        array_names = list(temp_mesh.point_arrays)
        if not array_names:
            temp_mesh = temp_mesh.cell_data_to_point_data()
            array_names = list(temp_mesh.point_arrays)
        print(array_names)
        for temp_array_name in array_names:
            # The consequences of building this without a proper road map
            new_array_name = temp_array_name.replace("ESca1", "")
            new_array_name = new_array_name.replace("_original_", "_")
            new_array_name = new_array_name.replace("_smooth_", "_")
            new_array_name = new_array_name.replace("_mean_mean_", "_")
            new_array_name = new_array_name.replace(
                "_max_normalized_mean_", "_max_normalized_"
            )
            new_array_name = new_array_name.replace("_max_normalized_", "_max_norm_")
            new_array_name = new_array_name.replace("_std_", "_standard_dev_")
            new_array_name = new_array_name.replace("DA_val01_", "DA_")
            for scalar in scalars:
                new_array_name = new_array_name.replace(f"{scalar}_mean_", f"{scalar}_")
            new_array_name = f"{new_array_name}_{name_match}"
            if new_array_name[0] == "_":
                new_array_name = f"{new_array_name[1:]}"
            print(f"              {new_array_name}")
            temp_array = temp_mesh[temp_array_name]
            vtk_mesh[str(new_array_name)] = temp_array
    out_name = out_name.replace(".vtk", "")
    out_name = out_name.replace("__", "_")
    vtk_mesh.save(f"{out_name}.vtk")


def gather_multiscalar_vtk(
    input_mesh: str,
    out_name: str,
    name_match: Union[str, None],
):
    vtk_mesh = pv.read(input_mesh)
    vtk_mesh.clear_arrays()
    consolidate_vtks = glob.glob("*.vtk")
    if not name_match == None:
        consolidate_vtks = [
            item for item in consolidate_vtks if f"{name_match}" in consolidate_vtks
        ]

    for consolidate in consolidate_vtks:
        print(f"\n Consolidating {consolidate}")
        temp_mesh = pv.read(consolidate)
        array_names = list(temp_mesh.point_arrays)
        for temp_array_name in array_names:
            # The consequences of building this without a proper road map
            new_array_name = temp_array_name.replace("ESca1", "")
            new_array_name = new_array_name.replace("_original_", "_")
            new_array_name = new_array_name.replace("_smooth_", "_")
            new_array_name = new_array_name.replace("BVTV", "BVTV_mean_")
            new_array_name = new_array_name.replace("DA", "DA_mean_")
            new_array_name = new_array_name.replace("_mean_mean_", "_mean_")
            new_array_name = new_array_name.replace(
                "_max_normalized_mean_", "_max_normalized_"
            )
            new_array_name = new_array_name.replace("_max_normalized_", "_max_norm_")
            new_array_name = new_array_name.replace("_std_", "_standard_dev_")
            new_array_name = new_array_name.replace("DA_val01_", "DA_")
            new_array_name = new_array_name.replace("__", "_")
            new_array_name = f"{new_array_name}"
            if new_array_name[0] == "_":
                new_array_name = f"{new_array_name[1:]}"
            if new_array_name[-1] == "_":
                new_array_name = f"{new_array_name[:-1]}"

            print(f"              {new_array_name}")
            temp_array = temp_mesh[temp_array_name]
            vtk_mesh[str(new_array_name)] = temp_array
    out_name = out_name.replace(".vtk", "")
    out_name = out_name.replace("__", "_")
    vtk_mesh.save(f"{out_name}.vtk")
