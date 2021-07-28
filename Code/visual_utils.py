import os
import re
import sys
import glob
import pathlib
import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib import cm
from matplotlib.colors import rgb2hex
from PyPDF2 import PdfFileWriter, PdfFileReader
from scipy.spatial.transform import Rotation as R
from matplotlib.colors import LinearSegmentedColormap


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


def vtk_celldata_to_pointdata(inputMesh):
    """
    Function to return a point cloud and scalars from a case file.
    Adjusted from https://gist.github.com/mrklein/44a392a01fa3e0181972
    :param inputMesh:
    :return:
    """
    start = timer()
    print(f"Averaging cell values around the points...")

    # Load in the vtk cell to point data module and get the values for the first block
    c2p = vtk.vtkCellDataToPointData()
    c2p.SetInputData(inputMesh.GetBlock(0))
    c2p.Update()
    points_data = c2p.GetOutput()

    # Get the values for the point array
    point_scalars = points_data.GetPointData().GetArray(0)

    # Place the data into a numpy object for easy reading
    np_array = dsa.WrapDataObject(points_data)

    # Get the scalar value name for the first block
    array_name = np_array.PointData.keys()[0]
    print(f"Extracing {array_name}...")

    # Place it into a dataframe for easy use
    points = pd.DataFrame(np.array(np_array.GetPoints()))

    # Use list comprehension to rapidly grab the first float value in the stored tuple for each point (len(points)).
    np_scalars = pd.DataFrame(
        np.array([float(point_scalars.GetTuple(x)[0]) for x in range(0, len(points))])
    )

    # Concat the two dataframes along the y axis and rename the columns
    points_dataframe = pd.concat([points, np_scalars], axis=1)
    points_dataframe.columns = ["x", "y", "z", str(array_name)]
    _end_timer(start, message="Converting case to point cloud")
    return points_dataframe


def vtk_read_vtk(inputMesh):
    """
    Function to read in vtk mesh files.
    :param inputMesh:
    :return:
    """
    start = timer()
    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName(str(inputMesh))
    reader.Update()
    vtk_mesh = reader.GetOutput()
    print("\n")
    _vtk_print_mesh_info(vtk_mesh)
    print("\n")
    _end_timer(start, message="Reading in vtk file")
    return vtk_mesh


def _vtk_print_mesh_info(inputMesh):
    """
    Function to report basic information about a VTK mesh.
    :param inputMesh:
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


def _get_scalar_limits(
    input_mesh,
    scalar,
    scalar_list,
    divergent=True,
    remove_max_norm=True,
    average_scalar=False,
):
    """
    "Function to calculate the upper and lower bounds from a set of scalar points in a mesh."
    """
    current_list = [measure for measure in scalar_list if scalar in measure]
    if remove_max_norm:
        current_list = [
            measure for measure in current_list if "max_norm" not in measure
        ]
    if average_scalar:
        current_list = [
            measure for measure in current_list if "_standard_dev" not in measure
        ]
        current_list = [
            measure for measure in current_list if "_coef_var" not in measure
        ]
    array_min_max = _get_min_max(input_mesh=input_mesh, scalar_list=current_list)
    if divergent:
        limits = _divergent_limits(array_min_max)
        print(f"{scalar}:", limits)
    else:
        print(f"{scalar}:", array_min_max)
        print("\n")
        limits = array_min_max
    return limits


def _get_min_max(input_mesh, scalar_list):
    """
    Function to get the minimum and maximum values from a scalar
    """
    current_list = scalar_list
    array_min_max = []
    for points in current_list:
        current_array = np.array(input_mesh.get_data_range(arr_var=str(points)))
        try:
            current_array_min = float(
                current_array[(current_array < 100) & (current_array > -100)].min()
            )
        except ValueError:
            current_array_min = 0
        try:
            current_array_max = float(
                current_array[(current_array < 100) & (current_array > -100)].max()
            )
        except ValueError:
            current_array_max = 0
        array_min_max.append((current_array_min, current_array_max))
    array_min = np.array(array_min_max).min()
    array_max = np.array(array_min_max).max()
    array_min_max = [array_min, array_max]
    return array_min_max


def _divergent_limits(array_min_max):
    """
    Function to produce a signed set of limits.
    """
    array_min: float = np.array(array_min_max).min()
    array_max: float = np.array(array_min_max).max()
    if array_max < 0:
        array_max = -1 * array_min
    if array_min > 0:
        array_min = -1 * array_max

    if abs(array_min) <= array_max:
        limits = [
            (array_min + abs(array_min * 0.1)),
            (abs(array_min) - (abs(array_min) * 0.1)),
        ]
    else:
        array_min = -1 * array_max
        limits = [
            (array_min + abs(array_min * 0.1)),
            (abs(array_max) - (abs(array_max) * 0.1)),
        ]

    # if it is at least greater than 1, round it.
    if limits[1] >= 1:
        limits = [np.ceil(limits[0]), np.floor(limits[1])]
    else:
        limits = [
            (array_min + abs(array_min * 0.1)),
            (abs(array_max) - (abs(array_min) * 0.1)),
        ]
        if limits[1] >= 1:
            limits = [np.floor(limits[0]), np.floor(limits[1])]
    return limits


def merge_pdfs(pdf_directory, out_name, string_match="", output_directory=""):
    pdf_writer = PdfFileWriter()
    pdf_directory = pathlib.Path(pdf_directory)
    if string_match == "":
        pdf_list = glob.glob(str(pdf_directory.joinpath(f"*.pdf")))
    else:
        pdf_list = glob.glob(str(pdf_directory.joinpath(f"*{string_match}*.pdf")))
    if output_directory == "":
        output_directory = pathlib.Path.cwd()
    else:
        output_directory = pathlib.Path(output_directory)
    if not pdf_list:
        print("No pdf files found!")
    else:
        pdf_list.sort(key=natural_keys)
        out_name = out_name.replace(".pdf", "")
        output_file = output_directory.joinpath(f"{out_name}.pdf")
        print(f"Found {len(pdf_list)} pdf file. Combining now...")
        for pdf in pdf_list:
            pdf_reader = PdfFileReader(pdf)
            for page in range(pdf_reader.getNumPages()):
                # Add each page to the writer object
                pdf_writer.addPage(pdf_reader.getPage(page))

        # Write out the merged PDF
        with open(output_file, "wb") as out:
            pdf_writer.write(out)


def colormap_from_paraview(paraview_json):
    """
    Function to return the hex colormap from a paraview json. You can export this by choosing the colormap and hitting
    "Export". From there you can feed this into any matplotlib style grapher.
    """
    paraview_colormap = pd.read_json(paraview_json)
    red_points = paraview_colormap.RGBPoints[0][1::4]
    blue_points = paraview_colormap.RGBPoints[0][2::4]
    green_points = paraview_colormap.RGBPoints[0][3::4]
    rgb_points = zip(red_points, blue_points, green_points)
    rgb_list = []
    for rgb in rgb_points:
        rgb_list.append(rgb2hex(rgb))
    return rgb_list


def scalar_name_cleanup(input_mesh, replace_dict):
    renamed_mesh = input_mesh.copy()
    for s_array in renamed_mesh.point_arrays:
        for key, values in replace_dict.items():
            new_array = s_array.replace(key, values)
            try:
                renamed_mesh.rename_array(s_array, new_array)
            except:
                pass
    return renamed_mesh


def scalar_name_suffix(input_mesh, suffix):
    renamed_mesh = input_mesh.copy()
    for s_array in renamed_mesh.point_arrays:
        new_array = f"{s_array}{suffix}"
        new_array = new_array.replace(f"{suffix}{suffix}", f"{suffix}")
        renamed_mesh.rename_array(s_array, new_array)
    return renamed_mesh


def subset_scalar_list(
    scalar_list,
    scalar,
    scalar_string="",
    remove_strings=False,
    stringent=False,
    verbose=False,
):
    if scalar_string != "":
        try:
            full_list = [
                measure
                for measure in scalar_list
                if scalar in measure and scalar_string in measure
            ]
        except:
            full_list = False
        if not full_list:
            if stringent:
                if verbose:
                    print(f"Couldn't subset {scalar} scalars by {scalar_string}")
            return False
    else:
        full_list = [measure for measure in scalar_list if scalar in measure]

    if len(full_list) == 0:
        if verbose:
            print(f"Couldn't subset {scalar} scalars")
            return False
    else:
        if not remove_strings:
            return full_list
        elif type(remove_strings) == list:
            for remove in remove_strings:
                full_list = [
                    list_item for list_item in full_list if remove not in list_item
                ]
        elif type(remove_strings) == str:
            full_list = [
                list_item for list_item in full_list if remove_strings not in list_item
            ]
        subset_list = full_list
        return subset_list


def _get_analytical_lists(scalar_list, scalar):
    average_list = subset_scalar_list(
        scalar_list=scalar_list,
        scalar=str(scalar),
        scalar_string="",
        remove_strings=["_coef_var", "_standard_dev", "_HDI_range", "_posterior_sd"],
        stringent=False,
        verbose=True,
    )

    coef_list = subset_scalar_list(
        scalar_list=scalar_list,
        scalar=str(scalar),
        scalar_string="_coef_var",
        remove_strings=["_standard_dev"],
        stringent=True,
        verbose=False,
    )
    # TODO fix the nesting in scalar_string for subset_scalar_string
    std_list = subset_scalar_list(
        scalar_list=scalar_list,
        scalar=str(scalar),
        scalar_string="_standard_dev",
        remove_strings=False,
        stringent=True,
        verbose=False,
    )

    hdi_list = subset_scalar_list(
        scalar_list=scalar_list,
        scalar=str(scalar),
        scalar_string="_HDI_range",
        remove_strings=["_standard_dev", "_coef_var"],
        stringent=True,
        verbose=False,
    )

    sd_list = subset_scalar_list(
        scalar_list=scalar_list,
        scalar=str(scalar),
        scalar_string="_posterior_sd",
        remove_strings=["_standard_dev"],
        stringent=True,
        verbose=False,
    )
    return {
        "Average": average_list,
        "Coeff. Var.": coef_list,
        "Std. Dev.": std_list,
        "HDI": hdi_list,
        "Sd": sd_list,
    }


def get_scalar_screens(
    input_mesh,
    scalars=["BVTV", "DA"],
    limits=[0.0, 0.50],
    consistent_limits=True,
    n_of_bar_txt_portions=11,
    output_type="png",
    from_bayes=False,
    scale_without_max_norm=True,
    foot_bones=False,
):
    scalar_color_dict_10 = _generate_color_dict(n_bins=10)
    scalar_color_dict_256 = _generate_color_dict(n_bins=256)
    for scalar in scalars:
        scalar_list = list(input_mesh.point_arrays)
        analytical_dict = _get_analytical_lists(scalar_list, scalar)
        for key, values in analytical_dict.items():
            if values:
                print(key)
                if consistent_limits:
                    range_limits = _get_scalar_limits(
                        input_mesh=input_mesh,
                        scalar=scalar,
                        scalar_list=values,
                        divergent=False,
                        remove_max_norm=scale_without_max_norm,
                        average_scalar=False,
                    )
                if not consistent_limits:
                    for value in values:
                        range_max = float(input_mesh.point_arrays[str(value)].max())
                        range_min = float(input_mesh.point_arrays[str(value)].min())
                        range_limits = [range_min, range_max]

                if key == "Average":
                    if limits:
                        range_limits = limits
                    if scalar in ["BVTV", "CtTh", "BSBV", "Tb_N"]:
                        color_map = scalar_color_dict_256[str(scalar)]
                    else:
                        color_map = scalar_color_dict_10[str(scalar)]
                else:
                    color_map = scalar_color_dict_10[str(key)]

                for value in values:
                    if "max_norm" not in value:
                        scalar_type = f"{key}"
                    else:
                        scalar_type = f"max norm. {key}"
                        if key == "Average":
                            range_limits = [0.0, 1.0]
                    generate_plot(
                        input_mesh=input_mesh,
                        scalar=scalar,
                        scalar_value=value,
                        scalar_type=scalar_type,
                        colormap=color_map,
                        limits=range_limits,
                        n_of_bar_txt_portions=n_of_bar_txt_portions,
                        output_type=output_type,
                        from_bayes=from_bayes,
                        foot_bones=foot_bones,
                    )


def generate_plot(
    input_mesh,
    scalar,
    scalar_value,
    scalar_type,
    colormap,
    limits=[0, 0.50],
    n_of_bar_txt_portions=11,
    output_type="png",
    from_bayes=False,
    foot_bones=False,
):
    output_choices = ["svg", "eps", "ps", "pdf", "tex"]
    scalar = scalar
    if not from_bayes:
        group = scalar_value.split("_", 1)[0]
        scalar_name = scalar_value
    else:
        name_value = scalar_value.replace(f"{scalar}", "")
        name_value = name_value.replace("standard_dev_", "")
        name_value = name_value.replace("_max_norm_", "")
        name_value = name_value.replace("_posterior_mean", "")
        name_value = name_value.replace("_posterior_sd", "")
        name_value = name_value.replace("_HDI_range", "")
        group = name_value
        if group[0] == "_":
            group = group[1:]
        group = group.replace("_", " ")
        scalar_name = scalar_value

    # Set up the arguments to be passed to pyvista
    vel_dargs = dict(
        scalars=f"{scalar_name}",
        clim=limits,
        cmap=colormap,
        log_scale=False,
        reset_camera=False,
    )
    f_size = 15

    sargs = dict(
        title_font_size=1,
        label_font_size=25,
        shadow=False,
        n_labels=n_of_bar_txt_portions,
        vertical=True,
        position_x=0.05,
        position_y=0.2,
        height=0.65,
        width=0.1,
        italic=False,
        fmt="%.2f",
        font_family="arial",
    )

    plotter = pv.Plotter(shape=(3, 3), window_size=(3840, 2160))
    plotter.enable_parallel_projection()
    plotter.reset_camera()

    plotter.subplot(0, 0)
    plotter.enable_parallel_projection()
    plotter.add_mesh(input_mesh, **vel_dargs, scalar_bar_args=sargs)
    plotter.view_zy()
    current_pos = plotter.camera_position
    new_position = [current_pos[0], current_pos[1], (0.0, 0.0, 1.0)]
    plotter.camera_position = new_position
    if foot_bones:
        plotter.add_text("Plantar", font_size=f_size)
    else:
        plotter.add_text("Posterior", font_size=f_size)

    plotter.subplot(1, 0)
    plotter.enable_parallel_projection()
    plotter.add_mesh(input_mesh, **vel_dargs)
    plotter.enable_parallel_projection()
    plotter.view_yz()
    plotter.enable_parallel_projection()
    if foot_bones:
        plotter.add_text("Superior", font_size=f_size)
    else:
        plotter.add_text("Anterior", font_size=f_size)

    plotter.subplot(0, 1)
    plotter.add_mesh(input_mesh, **vel_dargs)
    plotter.view_zx()
    current_pos = plotter.camera_position
    r = R.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True)
    new_position = tuple(np.dot(np.array(r.as_matrix()), np.array(current_pos[0])))
    plotter.camera_position = [new_position, current_pos[1], (0.0, 0.0, 1.0)]
    plotter.add_text("Lateral", font_size=f_size)

    plotter.subplot(1, 1)
    plotter.add_mesh(input_mesh, **vel_dargs)
    plotter.enable_parallel_projection()
    plotter.view_xz()
    plotter.add_text("Medial", font_size=f_size)

    plotter.enable_parallel_projection()
    plotter.subplot(1, 2)
    plotter.add_mesh(input_mesh, **vel_dargs)
    plotter.view_xy()
    current_pos = plotter.camera_position
    r = R.from_euler("xyz", [0.0, 0.0, 90.0], degrees=True)
    new_position = tuple(np.dot(np.array(r.as_matrix()), np.array(current_pos[0])))
    plotter.camera_position = [new_position, current_pos[1], (0.0, 0.0, 1.0)]
    if foot_bones:
        plotter.add_text("Anterior", font_size=f_size)
    else:
        plotter.add_text("Proximal", font_size=f_size)

    plotter.subplot(0, 2)
    plotter.add_mesh(input_mesh, **vel_dargs)
    plotter.enable_parallel_projection()
    plotter.view_yx()
    if foot_bones:
        plotter.add_text("Posterior", font_size=f_size)
    else:
        plotter.add_text("Distal", font_size=f_size)
    plotter.add_text(
        f"{group} {scalar} {scalar_type}",
        position="upper_right",
        font_size=25,
        font=None,
        shadow=False,
        name=None,
        viewport=False,
    )

    plotter.reset_camera()

    # Midslices
    slc_x = input_mesh.slice(normal="x")
    slc_y = input_mesh.slice(normal="y")
    slc_z = input_mesh.slice(normal="z")

    plotter.subplot(2, 0)
    plotter.add_mesh(slc_x, **vel_dargs, scalar_bar_args=sargs)
    plotter.view_yz()
    plotter.add_text("X-midplane", font_size=f_size)

    plotter.subplot(2, 1)
    plotter.add_mesh(slc_y, **vel_dargs)
    plotter.view_xz()
    plotter.add_text("Y-midplane", font_size=f_size)

    plotter.subplot(2, 2)
    plotter.add_mesh(slc_z, **vel_dargs)
    plotter.view_yx()
    plotter.add_text("Z-midplane", font_size=f_size)
    # Set up a name for the screen shot
    screen_shot = f"{scalar}_{group}_{scalar_type}_screen_shot"
    screen_shot = screen_shot.replace(" ", "_")
    screen_shot = screen_shot.replace(".", "_")
    screen_shot = screen_shot.replace("__", "_")
    screen_shot = screen_shot.replace("__", "_")

    if output_type in output_choices:
        plotter.save_graphic(filename=f"{screen_shot}.{output_type}")
    else:
        plotter.show(interactive=False, screenshot=f"{screen_shot}.png")


def get_ttest_screens(
    input_mesh,
    scalars: list = ["BVTV", "DA"],
    limits: list = [],
    estimate_limits: bool = True,
    n_of_bar_txt_portions: int = 11,
    output_type: str = "png",
):
    stats_mesh = input_mesh
    scalar_color_dict_10 = _generate_color_dict(n_bins=10)
    color_map = scalar_color_dict_10["stats"]
    stats_colors = color_map
    array_list = list(stats_mesh.point_arrays)

    for scalar in scalars:
        scalar_list = [measure for measure in array_list if scalar in measure]

        if estimate_limits:
            limits = _get_scalar_limits(
                input_mesh=stats_mesh,
                scalar=scalar,
                scalar_list=scalar_list,
                divergent=True,
                remove_max_norm=False,
            )
        else:
            limits = list(limits)

        for current_points in scalar_list:
            generate_stats_plot(
                input_mesh=stats_mesh,
                scalar=scalar,
                scalar_value=current_points,
                scalar_type="T-Score",
                colormap=stats_colors,
                limits=limits,
                n_of_bar_txt_portions=n_of_bar_txt_portions,
                output_type=output_type,
            )


def get_bayes_thresh_screens(
    input_mesh,
    scalars=["BVTV", "DA"],
    limits=[-1, 1],
    estimate_limits=True,
    n_of_bar_txt_portions=11,
    output_type="png",
    foot_bones=False,
):
    stats_mesh = input_mesh
    scalar_color_dict_10 = _generate_color_dict(n_bins=10)
    scalar_list = list(stats_mesh.point_arrays)
    for scalar in scalars:
        subset_list = subset_scalar_list(
            scalar_list=scalar_list,
            scalar=str(scalar),
            scalar_string="",
            remove_strings=["_coef_var", "_standard_dev", "_HDI_range"],
            stringent=False,
            verbose=True,
        )
        if type(limits) == list:
            limits = limits
        elif estimate_limits:
            if "_POS" in subset_list[0]:
                divergent = False
            else:
                divergent = True
            limits = _get_scalar_limits(
                input_mesh=stats_mesh,
                scalar=scalar,
                scalar_list=subset_list,
                divergent=divergent,
                remove_max_norm=False,
            )
        else:
            limits = list(limits)

        for current_points in subset_list:
            if "_POS" in current_points:
                scalar_type = "Prob. Superiority"
                color_map = "POS"
            else:
                scalar_type = "Cohen's D"
                color_map = "stats"
            if "max_norm" in current_points:
                scalar_type = f"max norm. {scalar_type}"
            generate_stats_plot(
                input_mesh=stats_mesh,
                scalar=scalar,
                scalar_value=current_points,
                scalar_type=scalar_type,
                colormap=scalar_color_dict_10[str(color_map)],
                limits=limits,
                n_of_bar_txt_portions=n_of_bar_txt_portions,
                output_type=output_type,
                foot_bones=foot_bones,
            )


def generate_stats_plot(
    input_mesh,
    scalar: str,
    scalar_value,
    scalar_type,
    limits: list = [-1.0, 1.0],
    n_of_bar_txt_portions: int = 11,
    colormap="hot",
    output_type: str = "png",
    foot_bones=False,
):
    output_choices = ["svg", "eps", "ps", "pdf", "tex"]
    scalar = scalar
    scalar_name = scalar_value
    name_value = scalar_value.replace(f"{scalar}", "")
    name_value = name_value.replace("_max_norm_", "")
    name_value = name_value.replace("_POS", "")
    name_value = name_value.replace("_CohenD", "")
    name_value = name_value.replace("__", "_")
    name_value = name_value.replace("__", "_")
    group = name_value
    if group[0] == "_":
        group = group[1:]
    if group[-1] == "_":
        group = group[:-1]
    group = group.replace("_", " ")

    limits = limits
    stats_mesh = input_mesh
    print(scalar_name)
    threshed = stats_mesh.threshold(
        value=(-100.0, 100.0), scalars=f"{scalar_name}", preference="points"
    )
    vel_dargs = dict(
        scalars=f"{scalar_name}",
        clim=limits,
        cmap=colormap,
        log_scale=False,
        reset_camera=False,
    )

    sargs = dict(
        title_font_size=1,
        label_font_size=25,
        shadow=False,
        n_labels=n_of_bar_txt_portions,
        vertical=True,
        position_x=0.05,
        position_y=0.2,
        height=0.65,
        width=0.1,
        italic=False,
        fmt="%.2f",
        font_family="arial",
    )

    wire_args = dict(style="wireframe", color="white", opacity=0.10)

    # Check to see if there are any significant values at all. If there aren't then plot just the wireframe
    if threshed.n_points == 0:
        threshed = stats_mesh
        vel_dargs = wire_args

    f_size = 15
    plotter = pv.Plotter(shape=(2, 3), window_size=(3840, 2160))
    plotter.enable_parallel_projection()
    plotter.reset_camera()

    plotter.subplot(0, 0)
    plotter.enable_parallel_projection()
    plotter.add_mesh(stats_mesh, **wire_args)
    plotter.add_mesh(threshed, **vel_dargs, scalar_bar_args=sargs)
    plotter.view_zy()
    current_pos = plotter.camera_position
    new_position = [current_pos[0], current_pos[1], (0.0, 0.0, 1.0)]
    plotter.camera_position = new_position
    if foot_bones:
        plotter.add_text("Plantar", font_size=f_size)
    else:
        plotter.add_text("Dorsal", font_size=f_size)

    plotter.subplot(1, 0)
    plotter.enable_parallel_projection()
    plotter.add_mesh(stats_mesh, **wire_args)
    plotter.add_mesh(threshed, **vel_dargs)
    plotter.view_yz()
    if foot_bones:
        plotter.add_text("Superior", font_size=f_size)
    else:
        plotter.add_text("Anterior", font_size=f_size)

    plotter.subplot(0, 1)
    plotter.enable_parallel_projection()
    plotter.add_mesh(stats_mesh, **wire_args)
    plotter.add_mesh(threshed, **vel_dargs)
    plotter.view_zx()
    current_pos = plotter.camera_position
    r = R.from_euler("xyz", [0.0, 0.0, 0.0], degrees=True)
    new_position = tuple(np.dot(np.array(r.as_matrix()), np.array(current_pos[0])))
    plotter.camera_position = [new_position, current_pos[1], (0.0, 0.0, 1.0)]
    plotter.enable_parallel_projection()
    plotter.add_text("Medial", font_size=f_size)

    plotter.subplot(1, 1)
    plotter.enable_parallel_projection()
    plotter.add_mesh(stats_mesh, **wire_args)
    plotter.add_mesh(threshed, **vel_dargs)
    plotter.view_xz()
    plotter.add_text("Lateral", font_size=f_size)

    plotter.subplot(0, 2)
    plotter.enable_parallel_projection()
    plotter.add_mesh(stats_mesh, **wire_args)
    plotter.add_mesh(threshed, **vel_dargs)
    plotter.view_yx()
    plotter.enable_parallel_projection()
    if foot_bones:
        plotter.add_text("Posterior", font_size=f_size)
    else:
        plotter.add_text("Distal", font_size=f_size)
    plotter.add_text(
        f"{group} {scalar} {scalar_type}",
        position="upper_right",
        font_size=25,
        font=None,
        shadow=False,
        name=None,
        viewport=False,
    )

    plotter.enable_parallel_projection()
    plotter.subplot(1, 2)
    plotter.add_mesh(stats_mesh, **wire_args)
    plotter.add_mesh(threshed, **vel_dargs)
    plotter.view_xy()
    current_pos = plotter.camera_position
    r = R.from_euler("xyz", [0.0, 0.0, 90.0], degrees=True)
    new_position = tuple(np.dot(np.array(r.as_matrix()), np.array(current_pos[0])))
    plotter.camera_position = [new_position, current_pos[1], (0.0, 0.0, 1.0)]
    if foot_bones:
        plotter.add_text("Anterior", font_size=f_size)
    else:
        plotter.add_text("Proximal", font_size=f_size)
    plotter.reset_camera()

    # Set up a name for the screen shot
    screen_shot = f"{scalar}_{name_value}_{scalar_type}_screen_shot"
    screen_shot = screen_shot.replace(" ", "_")
    screen_shot = screen_shot.replace(".", "_")
    screen_shot = screen_shot.replace("__", "_")
    screen_shot = screen_shot.replace("__", "_")
    if output_type in output_choices:
        plotter.save_graphic(filename=f"{screen_shot}.{output_type}")
    else:
        plotter.show(interactive=False, screenshot=f"{screen_shot}.png")


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


def _generate_color_dict(n_bins=10):
    n_bins: int = int(n_bins)
    # Set up custom colormap from paraview
    rainbow_desat = LinearSegmentedColormap.from_list(
        "rainbow_desat",
        [
            "#4747db",
            "#00005c",
            "#00ffff",
            "#008000",
            "#ffff00",
            "#ff6100",
            "#6b0000",
            "#e04d4d",
        ],
        N=n_bins,
    )

    linear_blue = LinearSegmentedColormap.from_list(
        "linear_blue",
        [
            "#f5fffa",
            "#d0f5e9",
            "#abedde",
            "#8ee6d7",
            "#7aded2",
            "#70d4cd",
            "#66ccc9",
            "#60c4c4",
            "#59b5ba",
            "#53a7b0",
            "#4d9ba8",
            "#3f8b9e",
            "#3d7e94",
            "#3a708a",
            "#3a6785",
            "#3b5e80",
            "#3a5278",
            "#384870",
            "#313c66",
            "#292f59",
            "#22204d",
        ],
        N=n_bins,
    )

    blue_orange_div = LinearSegmentedColormap.from_list(
        "blue_orange_divergent",
        [
            "#16014c",
            "#1d0673",
            "#1b0d82",
            "#0a0a8f",
            "#081999",
            "#0b2aa3",
            "#0e3ead",
            "#0e51b5",
            "#0d65bd",
            "#0a77c4",
            "#0889c9",
            "#089dcf",
            "#06b5d4",
            "#0dccd9",
            "#12dade",
            "#43e6dc",
            "#6cf0df",
            "#92f6d5",
            "#a8fad7",
            "#c3fadd",
            "#d3fae2",
            "#e9fcef",
            "#fffff8",
            "#fcfade",
            "#fdf8cd",
            "#fdf6b6",
            "#fcf4a4",
            "#faea82",
            "#f7df68",
            "#f2d252",
            "#edc647",
            "#e8b73c",
            "#e3a832",
            "#e09e2b",
            "#de8c28",
            "#d97925",
            "#d46922",
            "#cf581d",
            "#c94418",
            "#bd2f13",
            "#b02010",
            "#9e100b",
            "#8c0712",
            "#780417",
            "#66011a",
            "#300012",
        ],
        N=n_bins,
    )

    # To get a colormap from paraview
    # linear_green = colormap_from_paraview(paraview_json=r"D:\Desktop\linear_green.json")

    linear_green = LinearSegmentedColormap.from_list(
        "linear_green_Gr4L",
        [
            "#0e1c1f",
            "#132c2e",
            "#163b38",
            "#184740",
            "#1c5947",
            "#1d6647",
            "#1e7345",
            "#1e7d3e",
            "#1d8534",
            "#1c8c27",
            "#159615",
            "#1ca10d",
            "#36ad15",
            "#51b81d",
            "#6ec229",
            "#8ecc3d",
            "#aad64b",
            "#c8e065",
            "#e2eb88",
            "#f5f2ab",
            "#fffbe6",
        ],
        N=n_bins,
    )

    # cold_and_hot = colormap_from_paraview(r"D:\Desktop\Cold_and_hot.json")

    cold_and_hot = LinearSegmentedColormap.from_list(
        "cold_and_hot",
        ["#00ffff", "#0000ff", "#000080", "#ff0000", "#ffff00"],
        N=n_bins,
    )

    # purple_2_pink = colormap_from_paraview(r"D:\Desktop\purple2pink.json")
    purple_2_pink = LinearSegmentedColormap.from_list(
        "purple_2_pink",
        [
            "#000000",
            "#1e072d",
            "#2d0c49",
            "#3d1163",
            "#4d187c",
            "#5f238e",
            "#733096",
            "#873f99",
            "#9c4d9c",
            "#b25a9d",
            "#c8679d",
            "#dc75a4",
            "#e986b4",
            "#ee9cc4",
            "#ebb4d0",
            "#e6cdda",
            "#e2e2e2",
        ],
        N=n_bins,
    )

    # blue_2_green = colormap_from_paraview(r"D:\Desktop\erdc_blue2green.json")
    blue_2_green = LinearSegmentedColormap.from_list(
        "blue_2_green",
        [
            "#000000",
            "#0e0f36",
            "#101363",
            "#121a8a",
            "#1d28a4",
            "#273eab",
            "#28589f",
            "#277088",
            "#2c8575",
            "#3a995d",
            "#52ab42",
            "#71bb39",
            "#8ecb42",
            "#acda57",
            "#c9e886",
            "#e5f4c6",
            "#ffffff",
        ],
        N=n_bins,
    )

    # muted_blue_green = colormap_from_paraview(r"D:\Desktop\muted_blue_green.json")
    muted_blue_green = LinearSegmentedColormap.from_list(
        "muted_blue_green",
        [
            "#1c464d",
            "#214f57",
            "#265761",
            "#306775",
            "#3a7285",
            "#4a7e96",
            "#5e8dab",
            "#759ebf",
            "#96b6d9",
            "#b9d0f0",
            "#d4e1fa",
            "#e8eeff",
            "#fafbff",
            "#fefff2",
            "#fffffa",
            "#fafbff",
            "#f8fce3",
            "#eaf5d5",
            "#d5ebc5",
            "#c4e6bc",
            "#abd4a7",
            "#93c295",
            "#7fad85",
            "#648f6d",
            "#4b7355",
            "#365941",
            "#274732",
        ],
        N=n_bins,
    )

    bsbv_colors = blue_2_green
    tbn_colors = blue_2_green
    bvtv_colors = rainbow_desat
    coef_colors = cm.get_cmap(name="hot", lut=10)
    cort_colors = blue_orange_div
    da_colors = linear_blue
    hdi_colors = muted_blue_green
    spacing_colors = purple_2_pink
    standard_dev_colors = cm.get_cmap(name="PiYG_r", lut=10)
    stats_colors = cold_and_hot
    pos_colors = cm.get_cmap(name="viridis", lut=10)
    thickness_colors = linear_green

    scalar_color_dict = {
        "BVTV": bvtv_colors,
        "DA": da_colors,
        "CtTh": cort_colors,
        "BSBV": bsbv_colors,
        "Tb_Sp": spacing_colors,
        "Tb_N": tbn_colors,
        "Tb_Th": thickness_colors,
        "Coeff. Var.": coef_colors,
        "Std. Dev.": standard_dev_colors,
        "HDI": hdi_colors,
        "stats": stats_colors,
        "POS": pos_colors,
        "Sd": standard_dev_colors,
    }
    return scalar_color_dict


def remove_array(input_mesh, remove_arrays):
    """
    I couldn't find a pop method or something similar in pyvista.
    """
    remove_arrays: list = remove_arrays
    altered_mesh = input_mesh.copy()
    scalar_list = list(altered_mesh.point_arrays)
    for remove in remove_arrays:
        scalar_list = [keep for keep in scalar_list if remove not in keep]
    altered_mesh.clear_arrays()
    for scalar in scalar_list:
        altered_mesh[scalar] = input_mesh[scalar]
    return altered_mesh
