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
import time
import glob
import pycpd
import shutil
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from scipy.stats import variation
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from timeit import default_timer as timer


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
    print("Iteration: {}. Error: {}.".format(iteration, error))


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

    if outDir == "":
        transform_name = f"{name_in}_{str(tranform_type)}_transformations.csv"
        correspondence_name = (
            f"{name_in}_{str(tranform_type)}_points_correspondence.csv"
        )
    else:
        transform_dir = pathlib.Path(outDir).joinpath(str(tranform_type))
        transform_name = transform_dir.joinpath(
            f"{name_in}_{str(tranform_type)}_transformations.csv"
        )
        correspondence_name = transform_dir.joinpath(
            f"{name_in}_{str(tranform_type)}_points_correspondence.csv"
        )

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

    if outDir == "":
        outname = f"{name_in}_rigid_moving.csv"
        matrixname = f"{name_in}_rigid_correspondence_matrix.csv"

    else:
        rigid_dir = pathlib.Path(outDir).joinpath("rigid")
        outname = rigid_dir.joinpath(f"{name_in}_rigid_moving.csv")
        matrixname = rigid_dir.joinpath(f"{name_in}_rigid_correspondence_matrix.csv")

    np.savetxt(str(outname), reg.TY, delimiter=",", fmt="%f")

    if matrix != False:
        np.savetxt(str(matrixname), reg.P, delimiter=",", fmt="%f")

    get_transform(name_in, reg, "rigid", outDir=outDir)
    end = timer()
    end_time = end - start
    print(end_time)


def rigid_high_to_low(
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

    if outDir == "":
        outname = f"{name_in}_aligned_original.csv"
        matrixname = f"{name_in}_aligned_original_correspondence_matrix.csv"

    else:
        rigid_dir = pathlib.Path(outDir).joinpath("rigid")
        outname = rigid_dir.joinpath(f"{name_in}_aligned_original.csv")
        matrixname = rigid_dir.joinpath(
            f"{name_in}_aligned_original_correspondence_matrix.csv"
        )

    np.savetxt(str(outname), reg.TY, delimiter=",", fmt="%f")

    if matrix != False:
        np.savetxt(str(matrixname), reg.P, delimiter=",", fmt="%f")

    get_transform(name_in, reg, "rigid", outDir=outDir)
    end = timer()
    end_time = end - start
    print(end_time)


def affine(name_in, moving, fixed, iterations, tolerance, matrix=False):
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
    reg = pycpd.AffineRegistration(**{"X": fixed, "Y": moving})
    reg.max_iterations = iterations
    reg.tolerance = tolerance
    reg.register(text_update)
    start = timer()
    print(moving)
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
    get_transform(name_in, reg, "affine")
    end = timer()
    end_time = end - start
    print(end_time)


def deformable(name_in, moving, fixed, iterations, tolerance, matrix=False):
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
    reg = pycpd.DeformableRegistration(**{"X": fixed, "Y": moving})
    reg.max_iterations = iterations
    reg.tolerance = tolerance
    reg.register(text_update)
    start = timer()
    print(moving)
    print("\n")
    print("New matrix:")
    print(reg.TY)  # The registered matrix
    np.savetxt(name_in + "_deformable_moving.csv", reg.TY, delimiter=",", fmt="%f")
    if matrix == False:
        pass
    else:
        np.savetxt(
            name_in + "_deformable_correspondence_matrix.csv",
            reg.P,
            delimiter=",",
            fmt="%f",
        )
    get_transform(name_in, reg, "deformable")
    end = timer()
    end_time = end - start
    print(end_time)


# Uses scipy interpolate to map the original values to the registered points
def map_cloud(
    name_in, grid, points, original, canonical_geo, scalar, scalar_values, is_3d=True
):
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
    if points.shape[0] - to_map.shape[0] != 0:
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
            if is_3d != True:
                fout.write("\npart\n         1\ntria3\n")
            else:
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
            name_In + "_" + str(scalar) + "_deformed_mapped.csv", sep=",", index=False
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
    name_in, grid, points, original, canonical_geo, scalar, scalar_values, is_3d=True
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
    if points.shape[0] - to_map.shape[0] != 0:
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
            if is_3d != True:
                fout.write("\npart\n         1\ntria3\n")
            else:
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
            name_In + "_" + str(scalar) + "_deformed_mapped_max_normalized.csv",
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
def get_mean_case(mean_fileList, outname, canonical_geo, scalar, is_3d=True):
    """
    :param mean_fileList:
    :param outname:
    :param canonical_geo:
    :return:
    """
    # Open an empty list to be appeneded to
    np_array_list = []
    scalar = str(scalar)
    # Loop throug the file_list and read in the values
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
        np.savetxt(name + "_" + str(scalar) + ".csv", data, "%5.10f", delimiter=",")
    # Make sure we have a string for the output name
    outname = str(outname)
    # Stack the numpy arrays vertically (which changed from the last version)
    comb_np_array = np.vstack(np_array_list)
    # Convert this to a dataframe, and transpose it so the individual files are column-wise.
    scalar_array = pd.DataFrame(comb_np_array).T
    # Write out the matrix to a csv
    scalar_array.to_csv(outname + "_all_" + str(scalar) + "_values.csv", index=False)
    # Create a new dataframe with pandas and get the mean across the rows
    mean_scalar = pd.DataFrame(scalar_array.mean(axis=1, numeric_only=True))
    # Check to make sure the vertex values are the same
    print("number of vertex values:", mean_scalar.shape[0])
    # Write out the mean to a csv
    mean_scalar.to_csv(
        outname + "mean_" + str(scalar) + "_3d.csv", index=False, sep=" "
    )  # pd.mean_bin
    # Write the mean values to a case file, with the fout in ensigh gold format
    # the "newline" argument prevents an automatic return being written on windows
    with open("ESca1" + outname + "_original_average.Esca1", "w", newline="") as fout:
        # Case header defining the name of the scalar array
        col_name1 = "ESca1" + outname + "_original_average"
        # fout to write the header
        fout.write(col_name1)
        if is_3d != True:
            fout.write("\npart\n         1\ntria3\n")
        else:
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
        if is_3d != True:
            fout.write("\npart\n         1\ntria3\n")
        else:
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
        if is_3d != True:
            fout.write("\npart\n         1\ntria3\n")
        else:
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
    shutil.copy(canonical_geo, outname + "_original_average.geo")


# Function for the initial rigid rotation of points
def initial_rigid(
    name_in,
    moving,
    auto3d_dir="",
    auto3d_ref="trabecular",
    rotation_matrix="",
    origin="",
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
                str(name_in) + "_" + str(auto3d_ref) + "_rotation_matrix.txt"
            )

            rotation_matrix = pd.read_csv(
                rotation_file, header=None, delim_whitespace=True
            )
            # Check to make sure it's the right shape
            print(rotation_matrix.shape)
            if rotation_matrix.shape[0] == 4:
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
    print("\n")
    print("New matrix:")
    print(reg)  # The registered matrixfor f in initial_rigid_fileList:
    print("\n")
    np.savetxt(name_in + "_aligned_original.csv", reg, delimiter=",", fmt="%f")
    end = timer()
    end_time = end - start
    print(end_time, "\n")


def set_origin_to_zero(name_In, point_cloud):
    name_in = name_In
    print(point_cloud)
    print(point_cloud.shape)
    centroid = get_centroid(point_cloud)
    print("Centroid:", centroid, "\n")
    print("Setting centroid origin to ~0")
    centered = point_cloud - centroid
    print("\n")
    print("centroid:")
    new_centroid = get_centroid(centered)
    with np.printoptions(precision=3, suppress=True):
        print(new_centroid)
    np.savetxt(name_in + "_aligned_original.csv", centered, delimiter=",", fmt="%f")


def get_mean_case_groups(group_list, group_identifiers, bone):
    dictionary = dict(zip(group_list, group_identifiers))
    print(dictionary)
    for (key, values) in dictionary.items():
        group = list(values)
        group_list_bvtv = []
        group_list_da = []
        group_list_ctth = []
        for i in group:
            try:
                group_list_bvtv.extend(
                    glob.glob("*" + str(i) + "*BVTV_original_mapped.csv")
                )
            except:
                print(f"No members for BVTV {key} group\n.")

            try:
                group_list_da.extend(
                    glob.glob("*" + str(i) + "*DA_original_mapped.csv")
                )
            except:
                print(f"No members for DA {key} group\n.")

            try:
                group_list_ctth.extend(
                    glob.glob("*" + str(i) + "*CtTh_original_mapped.csv")
                )
            except:
                print(f"No members for CtTh {key} group\n.")

        print("\n{} BVTV list contains {} items".format(key, len(group_list_bvtv)))

        if len(group_list_bvtv) > 2:
            outname = "Mean_" + str(key) + "_" + str(bone) + "_BVTV"
            print(outname)
            get_mean_case(group_list_bvtv, str(outname), mean_geo[0], "BVTV")
        else:
            print("Not enough individuals to create a mean.\n")

        print("\n{} DA list contains {} items".format(key, len(group_list_da)))

        if len(group_list_da) > 2:
            outname = "Mean_" + str(key) + "_" + str(bone) + "_DA"
            get_mean_case(group_list_da, str(outname), mean_geo[0], "DA")

        else:
            print("Not enough individuals to create a mean.\n")

        print(f"\n{key} CtTh list contains {len(group_list_ctth)} items")

        if len(group_list_ctth) > 1:
            outname = "Mean_" + str(key) + "_" + str(bone) + "_CtTh"
            get_mean_case(
                group_list_ctth, str(outname), mean_geo[0], "CtTh", is_3d=False
            )

        else:
            print("Not enough individuals to create a mean.\n")


def get_mean_case_groups_max_normalized(group_list, group_identifiers, bone):
    dictionary = dict(zip(group_list, group_identifiers))
    print(dictionary)
    for (key, values) in dictionary.items():
        group = list(values)
        group_list_bvtv = []
        group_list_da = []
        group_list_ctth = []
        for i in group:
            try:
                group_list_bvtv.extend(
                    glob.glob("*" + str(i) + "*BVTV_original_mapped_max_normalized.csv")
                )
            except:
                print("No members for BVTV {} group\n.".format(str(key)))

            try:
                group_list_da.extend(
                    glob.glob("*" + str(i) + "*DA_original_mapped_max_normalized.csv")
                )
            except:
                print("No members for DA {} group\n.".format(str(key)))

            try:
                group_list_ctth.extend(
                    glob.glob("*" + str(i) + "*CtTh_original_mapped_max_normalized.csv")
                )
            except:
                print(f"No members for CtTh {key} group\n.")

        print("\n{} BVTV list contains {} items".format(key, len(group_list_bvtv)))

        if len(group_list_bvtv) > 2:
            outname = "Mean_" + str(key) + "_" + str(bone) + "_BVTV_max_normalized"
            print(outname)
            get_mean_case(group_list_bvtv, str(outname), mean_geo[0], "BVTV")
        else:
            print("Not enough individuals to create a mean.\n")

        print("\n{} DA list contains {} items".format(key, len(group_list_da)))

        if len(group_list_da) > 2:
            outname = "Mean_" + str(key) + "_" + str(bone) + "_DA_max_normalized"
            get_mean_case(group_list_bvtv, str(outname), mean_geo[0], "DA")

        else:
            print("Not enough individuals to create a mean.\n")

        print(f"\n{key} CtTh list contains {len(group_list_ctth)} items")

        if len(group_list_ctth) > 1:
            outname = "Mean_" + str(key) + "_" + str(bone) + "_CtTh"
            get_mean_case(
                group_list_ctth, str(outname), mean_geo[0], "CtTh", is_3d=False
            )

        else:
            print("Not enough individuals to create a mean.\n")


######################################
#    Begin the actual operations     #
######################################

# Define the directory with the point clouds and change to it.
# If on Linux/Mac use forward slashes
# dir = pathlib.Path(r"/gpfs/group/LiberalArts/default/tmr21_collab/RyanLab/Projects/nsf_human_variation/Point_cloud/Calcaneus")

# If on Windows use back slashes with an "r" in front to not that it should be read (double backslashes also work).
dir = pathlib.Path(r"Z:\RyanLab\Projects\AGuerra\Nmel\Point_Clouds\Cortical")

# Change to the directory with the point cloud files.
os.chdir(str(dir))

# The auto3d_gm_directory
auto3d_dir = pathlib.Path(
    r"Z:\RyanLab\Projects\AGuerra\Nmel\Point_Clouds\Cortical\Off\Tibiotarsus_results\aligned"
)

bone = "Radius_Dist"

point_distance = "1.75"

# Define the "average\canonical" point cloud to be registered to the original point clouds.
canonical = glob.glob("canonical*.csv")
print(canonical)

# Define the average\canonical geometry
mean_geo = glob.glob("canonical_Radius_Dist_cortical.geo")
print(mean_geo)

# Define the group names for subsetting
group_list = ["KWapes", "Anteaters", "OWM", "nonKWapes"]

# Define the identifying text for each of the members of the group
group1_list = ["51202", "51393", "51377", "201588", "51379", "54091"]
group2_list = ["23437", "262655", "61795", "211662", "23436", "100068", "133490"]
group3_list = [
    "82096",
    "Papio_ursinus_Papio",
    "80774",
    "82097",
    "43086",
    "28256",
    "34712",
    "52206",
    "52223",
    "34714",
    "169430",
    "89365",
]
group4_list = ["NF821349", "NF821350", "NF821282", "NF821211", "200898"]

# Create a list of lists for the identifiers
group_identifiers = [group1_list, group2_list, group3_list, group4_list]
# group_identifiers = [group_list[0], group_list[1], group_list[2]]

######################################
#       Rigid registrations          #
######################################


# Use glob to match text of a file name and build a list of the point clouds to be aligned
# initial_rigid_fileList = glob.glob("*BVTV_point_cloud_" + str(point_distance) + "mm.csv")
initial_rigid_fileList = glob.glob("*cortical*" + ".csv")

# Sort the file list by ascending order
initial_rigid_fileList.sort()

# Print to console for verification
print(initial_rigid_fileList)

# Print the length of the list to verify it is finding all the files
print(len(initial_rigid_fileList))

initial_rigid_fileList = initial_rigid_fileList[:30]

"""

#If you want to subset the list you can do so by selecting the elements in the list: 
rigid_fileList = rigid_fileList[5:12]

"""

# For the initial transform you can morph the original point clouds so they are all aligned in the same direction
# The canonical point cloud should also be facing the same way, so this should make registration much easier

# Use a for loop to process each point cloud file (i.e. f) in the list
for f in initial_rigid_fileList:
    auto3d_dir = auto3d_dir
    name_In = f.replace("_cortical.csv", "")
    print("Initial registration with", name_In)
    moving = pd.read_csv(f, sep=",")
    moving = moving.iloc[:, 0:3].values
    print(moving.shape)
    # Perform the rigid registration with 100 iterations or an error of 0.05
    initial_rigid(
        name_In, moving, auto3d_dir, auto3d_ref="", rotation_matrix="", origin=""
    )
    print("\n")


# Set the origin of the aligned point clouds to 0 so they are all centered:
aligned_fileList = glob.glob("*_aligned_original.csv")
print(aligned_fileList)
print(len(aligned_fileList))

# Use a for loop to process each point cloud file (i.e. f) in the list
for f in aligned_fileList:
    # auto3d_dir = r"Z:\RyanLab\Projects\NStephens\Ovis_aries\Medtool\Auto3dGM\Trabecular\AST\Results\Aligned_Shapes"
    name_In = f.replace("_aligned_original.csv", "")
    print("Initial registration with", name_In)
    moving = pd.read_csv(f, sep=",", header=None).values
    # moving = moving.iloc[:, 1:4].values
    print(moving.shape)
    # Perform the rigid registration with 100 iterations or an error of 0.05
    set_origin_to_zero(name_In, moving)
    print("\n")

# Use a for loop to process each point cloud file (i.e. f) in the list
#
#
#    For molars     #
#
#
# for f in initial_rigid_fileList:
#    name_In = f.replace(".smooth.case_DNE_point_cloud_1.00mm0.csv", "")
#    print("Initial registration with", name_In)
#    moving = pd.read_csv(f, sep=",")
#    moving = moving.iloc[:, 1:4].values
#    print(moving.shape)
#    #Perform the rigid registration with 100 iterations or an error of 0.05
#    set_origin_to_zero(name_In, moving)
#    print("\n")


# Use glob to match text of a file name and build a list of the point clouds to be aligned
rigid_fileList = glob.glob("*_aligned_original.csv")

# Sort the file list by ascending order
rigid_fileList.sort()

# Print to console for verification
print(rigid_fileList)

# Print the length of the list to verify it is finding all the files
print(len(rigid_fileList))

# Loop over the list and perform the rigid registration
for f in rigid_fileList:
    # Read in the "average" with pandas and turn it into a numpy array
    moving = pd.read_csv(canonical[0], sep=",", header=None)
    moving = moving.values
    print("Points in the moving cloud:\n                        ", len(moving))
    # Setup how the results will be written with the name
    name_In = f.replace(".csv", "")
    print("Rigid registration with", name_In)
    fixed = pd.read_csv(name_In + ".csv", header=None, sep=",").values
    # fixed = fixed.iloc[:,1:4].values
    print_fixed_points(fixed)
    print("\n")
    # Perform the rigid registration with 100 iterations or an error of 0.05
    rigid(name_In, moving, fixed, 100, 0.05)
    print("\n")


"""
If you aren't using the auto3dGm alignment, it is recommended to do an initial rigid alignment, which can be visually
verified and corrected in CloudCompare. The corrected point clouds can then be run through in a second rigid
alignment.  
"""
"""
rigid_fileList = glob.glob("_BVTV_point_cloud_1.75mm*.csv")
rigid_fileList.sort()
print(rigid_fileList)
print(len(rigid_fileList))

for f in rigid_fileList:
    moving = pd.read_csv(canonical[0], sep=",", header=None)
    moving = moving.iloc[:, 0:3].values
    print_moving_points(moving)
    name_In = f.replace(".csv", "")
    print("Initial rigid registration with", name_In)
    fixed = pd.read_csv(name_In + ".csv", header=None, sep=",").values
    fixed = fixed.iloc[:,1:4].values
    print_fixed_points(fixed)
    print("\n")
    rigid(name_In, moving, fixed, 100, 0.01)
    print("\n")



"""
# If needed perform a second rigid registration to the aligned original.
"""

rigid_fileList = glob.glob("_rigid_moving.csv")
rigid_fileList.sort()
print(rigid_fileList)
print(len(rigid_fileList))

for f in rigid_fileList:
    moving = pd.read_csv(f, header=None, sep=",")
    moving = moving.values
    print_moving_points(moving)
    name_In = f.replace("_rigid_moving.csv", "")
    print("Second rigid registration with", name_In)
    fixed = pd.read_csv(name_In + "_aligned_original.csv", header=None, sep=",").values
    #fixed = fixed.iloc[:,1:4].values
    print_fixed_points(fixed)
    print("\n")
    rigid(name_In, moving, fixed, 100, 0.01)
    print("\n")

"""
######################################
#       Affine registrations         #
######################################

"""
Do the same with the affine registration. The difference here is that the canonical has been changed, 
so now we register the output from the previous registration to the aligned original (e.g. name_In + rigid).
"""

affine_fileList = glob.glob("*_rigid_moving.csv")
affine_fileList.sort()
print(affine_fileList)
print(len(affine_fileList))

for f in affine_fileList:
    moving = pd.read_csv(f, header=None, sep=",")
    moving = moving.values
    print_moving_points(moving)
    name_In = f.replace("_aligned_original_rigid_moving.csv", "")
    print("Affine registration with", name_In)
    fixed = pd.read_csv(name_In + "_aligned_original.csv", header=None, sep=",").values
    # fixed = fixed.iloc[:, 1:4].values
    print_fixed_points(fixed)
    print("Fixed points", fixed.shape[0], "Dimensionality:", fixed.shape[1])
    print(fixed)
    print("\n")
    affine(name_In, moving, fixed, 200, 0.00005)
    print("\n")


######################################
#       Deformable registration      #
######################################

"""
Do the deformable registration, which can take a much longer time.
"""

deformable_fileList = glob.glob("*_affine_moving.csv")
deformable_fileList.sort()
print(deformable_fileList)
print(len(deformable_fileList))

for f in deformable_fileList:
    moving = pd.read_csv(f, header=None, sep=",")
    moving = moving.values
    print_moving_points(moving)
    name_In = f.replace("_affine_moving.csv", "")
    # name_In = f.replace("_affine_moving.csv", "")
    # name_In = f.replace("_deformable_moving.csv", "")
    print("Deformable registration with", name_In)
    fixed = pd.read_csv(name_In + "_aligned_original.csv", sep=",").values
    # fixed = fixed.iloc[:, 1:4].values
    print_fixed_points(fixed)
    deformable(name_In, moving, fixed, 100, 0.0000000001)
    print("\n")


######################################
#    Mapping and mesh generation     #
######################################

"""
Now we map the scalar values from the orginal to the corresponding points on the registered point cloud 
using the scipy linear interpolation. This reguires a grid (i.e. the registered x, y, z) and the scalar values
from the original point cloud. Because the index has remained the same between the aligned and original point cloud
the order has remained the same. These can then be mapped taking into acount the aligned point clouds x, y, z 
to the corresponding x, y, z, of the registered point cloud. 
"""

mapping_fileList = glob.glob("*_deformable_moving.csv")
mapping_fileList.sort()
print(mapping_fileList)
print(len(mapping_fileList))

# Read in the canonical so the geometry of the mesh can be applied to the new case files.
# canonical = glob.glob("canonical*2.00mm*.csv")
canonical = canonical
print(canonical)

scalar_means = pd.DataFrame()

for f in mapping_fileList:
    grid = pd.read_csv(f, header=None, sep=",")
    print("Length of grid in", len(grid))
    name_In = f.replace("_deformable_moving.csv", "")
    # name_In = f.replace("_affine_moving.csv", "")
    print("Values for", name_In, "being mapped to", f)
    print("\n")
    to_map = pd.read_csv(name_In + "_aligned_original.csv", header=None, sep=",")
    points = to_map.iloc[:, 0:3]
    scalar_values = pd.read_csv(name_In + "_cortical.csv", sep=",")
    scalar_values = scalar_values.iloc[:, 3:4]
    Ct_mean = pd.DataFrame(scalar_values.mean(axis=0))
    Ct_mean.columns = [str(name_In) + "_CtTh"]
    original = canonical[0]
    canonical_geo = canonical[0].replace("_pointcloud.csv", ".geo")
    map_cloud(
        name_In,
        grid,
        points,
        original,
        canonical_geo,
        "CtTh",
        scalar_values,
        is_3d=False,
    )
    map_cloud_max_normalized(
        name_In,
        grid,
        points,
        original,
        canonical_geo,
        "CtTh",
        scalar_values,
        is_3d=False,
    )
    print("\n")
map_cloud
scalar_means = pd.DataFrame(
    np.sort(scalar_means.values, axis=0),
    index=scalar_means.index,
    columns=scalar_means.columns,
)
scalar_means = scalar_means.dropna()
scalar_means.to_csv("Scalar_mean_prior_to_mapping.csv", index=False)

# Visualize the amount of movement in the point clouds
rigid_fileList = glob.glob("*_aligned_original_rigid_moving.csv")
print(rigid_fileList)
print(len(rigid_fileList))

# Create and empty dataframe to append distance values to
all_distances = pd.DataFrame()

# Loop through the rigid file list and load the corresponding deformable point cloud
# Then calculate the distance between the rows for each point and write out a new point cloud

for f in rigid_fileList:
    name_In = f.replace("_aligned_original_rigid_moving.csv", "")
    print(name_In)
    rigid_df = pd.read_csv(f, header=None)
    deformable = pd.read_csv(str(name_In) + "_deformable_moving.csv", header=None)
    distance = pd.DataFrame(get_distance(rigid_df, deformable))
    all_distances = pd.concat([all_distances, distance], axis=1)
    rigid_df = pd.concat([rigid_df, distance], axis=1)
    rigid_df.columns = ["x", "y", "z", "distance"]
    print(rigid_df)
    rigid_df.to_csv(str(name_In) + "_registration_movement.csv")

# Calculate the mean distance for each point of the point cloud across the sample
mean_distance = all_distances.mean(axis=1)

# Find the canonical point cloud
canonical = glob.glob("*canonical*.csv")

# Read it in via pandas and concat it to a new column
canonical = pd.read_csv(canonical[0], header=None)
canonical["Mean_deformation"] = mean_distance

canonical.columns = ["x", "y", "z", "Mean_deformation"]

canonical.to_csv("Mean_deformation.csv", index=False)


######################################
# Generating cohort and group means  #
######################################

"""
In case you forgot to save the geo file from the flipped "average" specimen 
df = pd.read_csv("mean_canid_trab_1mm0_flipped00.csv", sep=",")
#Collapses the columns into one which mimics the ascii .geo file 
df2 = pd.Series(df.values.ravel('F'))
"""


# Gather all values for the points mapped and average them
# Still need to make the original and deformed consistent

# dir = pathlib.Path(r"/gpfs/group/LiberalArts/default/tmr21_collab/RyanLab/Projects/nsf_human_variation/Point_cloud/Calcaneus_2mm")
# dir = pathlib.Path(r"Z:\RyanLab\Projects\NStephens\Canids\Point_cloud\Femur")
# os.chdir(dir)
mean_fileList_CtTh = glob.glob("*CtTh_original_mapped.csv")
print(mean_fileList_CtTh)
print(len(mean_fileList_CtTh))

# Get the mean for the entire dataset
get_mean_case(
    mean_fileList_CtTh, "Mean_" + str(bone) + "_CtTh", mean_geo[0], "CtTh", is_3d=False
)

mean_fileList_CtTh_max_normalized = glob.glob(
    "*CtTh_original_mapped_max_normalized.csv"
)
print(mean_fileList_CtTh_max_normalized)
print(len(mean_fileList_CtTh_max_normalized))

get_mean_case(
    mean_fileList_CtTh_max_normalized,
    "Mean_" + str(bone) + "_CtTh_max_normalized",
    mean_geo[0],
    "CtTh",
)

get_mean_case_groups(group_list, group_identifiers, bone)

get_mean_case_groups_max_normalized(group_list, group_identifiers, bone)
