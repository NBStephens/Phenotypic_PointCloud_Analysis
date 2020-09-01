'''


from matplotlib.colors import ListedColormap
# https://kitware.github.io/paraview-docs/latest/python/_modules/paraview/_colorMaps.html
linear_blue = [[0.960784, 1.000000, 0.980392], [0.815686, 0.960784, 0.913725],  [0.670588, 0.929412, 0.870588],
                [0.556863, 0.901961, 0.843137], [0.478431, 0.870588, 0.823529], [0.439216, 0.831373, 0.803922],
                [0.400000, 0.800000, 0.788235], [0.376471, 0.768627, 0.768627], [0.349020, 0.709804, 0.729412],
                [0.325490, 0.654902, 0.690196], [0.301961, 0.607843, 0.658824], [0.247059, 0.545098, 0.619608],
                [0.239216, 0.494118, 0.580392], [0.227451, 0.439216, 0.541176], [0.227451, 0.403922, 0.521569],
                [0.231373, 0.368627, 0.501961], [0.227451, 0.321569, 0.470588], [0.219608, 0.282353, 0.439216],
                [0.192157, 0.235294, 0.400000], [0.160784, 0.184314, 0.349020], [0.133333, 0.125490, 0.301961]]

lb_list = []
for blue in linear_blue:
    lb_list.append(rgb2hex(blue))


'''

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

def get_scalar_screens(input_mesh, scalars=["BVTV", "DA"], limits=[0.0, 0.50], consistent_limits=True, n_of_bar_txt_portions=11, output_type="png"):
    BVTV_colors = rainbow_desat
    DA_colors = cm.get_cmap(linear_blue)
    Coef_colors = cm.get_cmap(name="hot", lut=10)
    Standard_dev_colors = cm.get_cmap(name="PiYG_r", lut=10)

    scalar_color_dict = {"BVTV": BVTV_colors, "DA": DA_colors, "coef": Coef_colors, "std": Standard_dev_colors}
    #Clean up an stupid naming thing I've done....
    for s_array in input_mesh.point_arrays:
        new_array = s_array.replace("DA_", "_DA_")
        new_array = new_array.replace("BVTV_", "_BVTV_")
        new_array = new_array.replace("__", "_")
        input_mesh.rename_array(s_array, new_array)

    for scalar in scalars:
        scalar_list = list(input_mesh.point_arrays)

        full_list = [measure for measure in scalar_list if "_average" and str(scalar) in measure]
        average_list = [list_item for list_item in full_list if "_coef_var" not in list_item]
        average_list = [list_item for list_item in average_list if "_standard_dev" not in list_item]

        coef_list = [measure for measure in full_list if "_coef_var" in measure]
        std_list = [measure for measure in full_list if "_standard_dev" in measure]

        coef_scalar = f"{scalar}_coef_var"
        coef_limits = _get_scalar_limits(input_mesh=input_mesh, scalar=coef_scalar, divergent=False)

        std_scalar = f"{scalar}_standard_dev"
        std_limits = _get_scalar_limits(input_mesh=input_mesh, scalar=std_scalar, divergent=False)

        for average_scalar in average_list:
            generate_plot(input_mesh=input_mesh,
                          scalar=scalar,
                          scalar_value=average_scalar,
                          scalar_type="average",
                          colormap=scalar_color_dict[str(scalar)],
                          limits=limits,
                          n_of_bar_txt_portions=n_of_bar_txt_portions,
                          output_type=output_type)

        for coef in coef_list:
            if not consistent_limits:
                range_max = float(input_mesh.point_arrays[str(coef)].max())
                range_min = float(input_mesh.point_arrays[str(coef)].min())
                coef_limits = [range_min, range_max]

            generate_plot(input_mesh=input_mesh,
                          scalar=scalar,
                          scalar_value=coef,
                          scalar_type="Coeff. Var.",
                          colormap=scalar_color_dict["coef"],
                          limits=coef_limits,
                          n_of_bar_txt_portions=11,
                          output_type=output_type)

        for std in std_list:
            if not consistent_limits:
                range_max = float(input_mesh.point_arrays[str(std)].max())
                range_min = float(input_mesh.point_arrays[str(std)].min())
                std_limits = [range_min, range_max]

            generate_plot(input_mesh=input_mesh,
                          scalar=scalar,
                          scalar_value=std,
                          scalar_type="Std. Dev.",
                          colormap=scalar_color_dict["std"],
                          limits=std_limits,
                          n_of_bar_txt_portions=11,
                          output_type=output_type)


def generate_plot(input_mesh, scalar, scalar_value, scalar_type, colormap, limits=[0, 0.50],n_of_bar_txt_portions=11, output_type="png"):

        scalar = scalar
        group = scalar_value.split("_", 1)[0]
        scalar_name = scalar_value

        #Set up the arguments to be passed to pyvista
        vel_dargs = dict(scalars=f"{scalar_name}", clim=limits, cmap=colormap,
                         log_scale=False, reset_camera=False)
        f_size = 15

        sargs = dict(title_font_size=1, label_font_size=25, shadow=False, n_labels=n_of_bar_txt_portions, vertical=True,
                     position_x=0.05, position_y=0.2,
                     height=0.65, width=0.1, italic=False, fmt="%.2f", font_family="arial")

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
        plotter.add_text("Dorsal", font_size=f_size)

        plotter.subplot(1, 0)
        plotter.enable_parallel_projection()
        plotter.add_mesh(input_mesh, **vel_dargs)
        plotter.enable_parallel_projection()
        plotter.view_yz()
        plotter.enable_parallel_projection()
        plotter.add_text("Anterior", font_size=f_size)


        plotter.subplot(0, 1)
        plotter.add_mesh(input_mesh, **vel_dargs)
        plotter.view_zx()
        current_pos = plotter.camera_position
        r = R.from_euler('xyz', [0.0, 0.0, 0.0], degrees=True)
        new_position = tuple(np.dot(np.array(r.as_matrix()), np.array(current_pos[0])))
        plotter.camera_position = [new_position, current_pos[1], (0.0, 0.0, 1.0)]
        plotter.add_text("Medial", font_size=f_size)

        plotter.subplot(1, 1)
        plotter.add_mesh(input_mesh, **vel_dargs)
        plotter.enable_parallel_projection()
        plotter.view_xz()
        plotter.add_text("Lateral", font_size=f_size)


        plotter.subplot(0, 2)
        plotter.add_mesh(input_mesh, **vel_dargs)
        plotter.enable_parallel_projection()
        plotter.view_yx()
        plotter.add_text("Distal", font_size=f_size)
        plotter.add_text(f"{group} {scalar} {scalar_type}", position='upper_right', font_size=25,
                         font=None, shadow=False, name=None, viewport=False)


        plotter.subplot(1, 2)
        plotter.add_mesh(input_mesh, **vel_dargs)
        plotter.enable_parallel_projection()
        plotter.view_yx()
        current_pos = plotter.camera_position
        r = R.from_euler('xyz', [0.0, 180.0, 0.0], degrees=True)
        new_position = tuple(np.dot(np.array(r.as_matrix()), np.array(current_pos[0])))
        new_position = [new_position, current_pos[1], current_pos[2]]
        plotter.camera_position = new_position
        plotter.add_text("Proximal", font_size=f_size)

        plotter.reset_camera()

        # Midslices
        slc_x = input_mesh.slice(normal='x')
        slc_y = input_mesh.slice(normal='y')
        slc_z = input_mesh.slice(normal='z')

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
        screen_shot = f"{scalar}_{group}_{scalar_type}_screen_shot.png"
        if output_type == "pdf":
            plotter.save_graphic(filename=f"{screen_shot[:-3]}pdf")
        else:
            plotter.show(interactive=False, use_panel=False, screenshot=str(screen_shot))

def _get_scalar_limits(input_mesh, scalar, divergent=True):
    """
    "Function to calculate the upper and lower bounds from a set of scalar points in a mesh."
    """
    stats_mesh = input_mesh
    scalar_list = list(stats_mesh.point_arrays)
    current_list = [measure for measure in scalar_list if scalar in measure]
    array_min_max = []
    for points in current_list:
        current_array = stats_mesh.point_arrays[points]
        try:
            current_array_min = float(current_array[(current_array < 100) & (current_array > -100)].min())
        except ValueError:
            current_array_min = 0
        try:
            current_array_max = float(current_array[(current_array < 100) & (current_array > -100)].max())
        except ValueError:
            current_array_max = 0
        array_min_max.append((current_array_min, current_array_max))

    array_min = np.array(array_min_max).min()
    array_max = np.array(array_min_max).max()

    # Make sure they always have negative and positive signs
    if divergent:
        if array_max < 0:
            array_max = -1 * array_min

        if array_min > 0:
            array_min = -1 * array_max

        if abs(array_min) <= array_max:
            limits = [(array_min + abs(array_min * 0.1)), (abs(array_min) - (abs(array_min) * 0.1))]

        else:
            array_min = -1 * array_max
            limits = [(array_min + abs(array_min * 0.1)), (abs(array_max) - (abs(array_max) * 0.1))]

        #if it is at least greater than 1, round it.
        if limits[1] >= 1:
            limits = [np.ceil(limits[0]), np.floor(limits[1])]
    else:
        limits = [(array_min + abs(array_min * 0.1)), (abs(array_max) - (abs(array_min) * 0.1))]
        if limits[1] >= 1:
            limits = [np.floor(limits[0]), np.floor(limits[1])]
    return limits

def get_ttest_screens(input_mesh, scalars=["BVTV", "DA"], estimate_limits=True, n_of_bar_txt_portions=11, output_type="png"):
    stats_mesh = input_mesh
    stats_colors = cm.get_cmap(cold_and_hot)
    array_list = list(stats_mesh.point_arrays)

    for scalar in scalars:
        scalar_list = [measure for measure in array_list if scalar in measure]

        if estimate_limits:
            limits = _get_scalar_limits(input_mesh=stats_mesh, scalar=scalar)
        else:
            limits = list(limits)

        for current_points in scalar_list:
            generate_stats_plot(input_mesh=stats_mesh,
                                scalar=scalar,
                                scalar_value=current_points,
                                scalar_type="T-Score",
                                colormap=stats_colors,
                                limits=limits,
                                n_of_bar_txt_portions=n_of_bar_txt_portions,
                                output_type=output_type)


def generate_stats_plot(input_mesh, scalar, scalar_value, scalar_type, limits=[-1.0, 1.0], n_of_bar_txt_portions=11, colormap="hot", output_type="png"):
    scalar = scalar
    group = scalar_value.split("_")
    vs_index = group.index("vs")
    group = [group[int(vs_index + -1)], group[int(vs_index + 1)]]
    scalar_name = scalar_value
    limits = limits
    stats_mesh = input_mesh
    print(scalar_name)
    threshed = stats_mesh.threshold(value=(-100, 100),
                                    scalars=f'{scalar_name}',
                                    invert=False,
                                    continuous=False,
                                    preference='point',
                                    all_scalars=False)


    vel_dargs = dict(scalars=f"{scalar_name}", clim=limits, cmap=colormap,
                     log_scale=False, reset_camera=False)

    sargs = dict(title_font_size=1, label_font_size=25, shadow=False, n_labels=n_of_bar_txt_portions, vertical=True,
                 position_x=0.05, position_y=0.2,
                 height=0.65, width=0.1, italic=False, fmt="%.2f", font_family="arial")

    wire_args = dict(style='wireframe', color='white', opacity=0.10)

    #Check to see if there are any significant values at all. If there aren't then plot just the wireframe
    if threshed.n_points is 0:
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
    plotter.add_text("Dorsal", font_size=f_size)

    plotter.subplot(1, 0)
    plotter.enable_parallel_projection()
    plotter.add_mesh(stats_mesh, **wire_args)
    plotter.add_mesh(threshed, **vel_dargs)
    plotter.view_yz()
    plotter.add_text("Anterior", font_size=f_size)

    plotter.subplot(0, 1)
    plotter.enable_parallel_projection()
    plotter.add_mesh(stats_mesh, **wire_args)
    plotter.add_mesh(threshed, **vel_dargs)
    plotter.view_zx()
    current_pos = plotter.camera_position
    r = R.from_euler('xyz', [0.0, 0.0, 0.0], degrees=True)
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
    plotter.add_text("Distal", font_size=f_size)
    plotter.add_text(f"{group[0]} vs {group[1]} {scalar} {scalar_type}", position='upper_right', font_size=25,
                     font=None, shadow=False, name=None, viewport=False)

    plotter.subplot(1, 2)
    plotter.enable_parallel_projection()
    plotter.add_mesh(stats_mesh, **wire_args)
    plotter.add_mesh(threshed, **vel_dargs)
    plotter.view_yx()
    current_pos = plotter.camera_position
    r = R.from_euler('xyz', [0.0, 180.0, 0.0], degrees=True)
    new_position = tuple(np.dot(np.array(r.as_matrix()), np.array(current_pos[0])))
    new_position = [new_position, current_pos[1], current_pos[2]]
    plotter.camera_position = new_position
    plotter.enable_parallel_projection()
    plotter.add_text("Proximal", font_size=f_size)

    # Set up a name for the screen shot
    screen_shot = f"{scalar}_{group[0]}_vs_{group[1]}{scalar_type}_screen_shot.png"
    if output_type == "pdf":
        plotter.save_graphic(filename=f"{screen_shot[:-3]}pdf")
    else:
        plotter.show(interactive=False, use_panel=False, screenshot=str(screen_shot))

import os
import sys
import pathlib
import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib import cm
from matplotlib.colors import rgb2hex
from scipy.spatial.transform import Rotation as R
from matplotlib.colors import LinearSegmentedColormap

directory = pathlib.Path(r"D:\Desktop\Canids\Point_cloud\Femur")
os.chdir(directory)

#Set the theme, default,  dark,
pv.set_plot_theme("default")

#Set up custom colormap from paraview
rainbow_desat = LinearSegmentedColormap.from_list('rainbow_desat',
                                                  ['#4747db', '#00005c', '#00ffff', '#008000',
                                                   '#ffff00', '#ff6100', '#6b0000', '#e04d4d'],
                                                  N=256)

linear_blue = LinearSegmentedColormap.from_list('linear_blue',
                                                ['#f5fffa', '#d0f5e9', '#abedde', '#8ee6d7', '#7aded2', '#70d4cd',
                                                 '#66ccc9', '#60c4c4', '#59b5ba', '#53a7b0', '#4d9ba8', '#3f8b9e',
                                                 '#3d7e94', '#3a708a', '#3a6785', '#3b5e80', '#3a5278', '#384870',
                                                 '#313c66', '#292f59', '#22204d'],
                                                N=10)

blue_orange_div = LinearSegmentedColormap.from_list('blue_orange_divergent',
                                                    ['#16014c',  '#1d0673', '#1b0d82', '#0a0a8f', '#081999', '#0b2aa3',
                                                     '#0e3ead', '#0e51b5', '#0d65bd', '#0a77c4', '#0889c9', '#089dcf',
                                                     '#06b5d4', '#0dccd9', '#12dade', '#43e6dc', '#6cf0df', '#92f6d5',
                                                     '#a8fad7', '#c3fadd', '#d3fae2', '#e9fcef', '#fffff8', '#fcfade',
                                                     '#fdf8cd', '#fdf6b6', '#fcf4a4', '#faea82', '#f7df68', '#f2d252',
                                                     '#edc647', '#e8b73c', '#e3a832', '#e09e2b', '#de8c28', '#d97925',
                                                     '#d46922', '#cf581d', '#c94418', '#bd2f13', '#b02010', '#9e100b',
                                                     '#8c0712', '#780417', '#66011a', '#300012'],
                                                    N=256)



#cold_and_hot = colormap_from_paraview(r"D:\Desktop\Cold_and_hot.json")

cold_and_hot = LinearSegmentedColormap.from_list('cold_and_hot',
                                                 ['#00ffff', '#0000ff', '#000080', '#ff0000', '#ffff00'],
                                                 N=10)



BVTV_colors = rainbow_desat
DA_colors = cm.get_cmap(linear_blue)
Coef_colors = cm.get_cmap(name="hot", lut=10)
Standard_dev_colors = cm.get_cmap(name="PiYG_r", lut=10)
cort_colors = cm.get_cmap(blue_orange_div)
stats_colors = cm.get_cmap(cold_and_hot, lut=10)

#Define the input name and read in to memory for visualization
mesh_name = "femur_consolidated_scalars.vtk"
input_mesh = pv.read(mesh_name)

#Output either a png or a pdf
get_scalar_screens(input_mesh=input_mesh,
                   scalars=["BVTV", "DA"],
                   limits=[0.0, 0.50],
                   consistent_limits=True,
                   n_of_bar_txt_portions=6,
                   output_type="pdf")

#If one scalar is being used or the range shoudl be somehting different
get_scalar_screens(input_mesh=input_mesh, scalars=["BVTV"], limits=[0.0, 0.60], consistent_limits=True, n_of_bar_txt_portions=6, output_type="pdf")
get_scalar_screens(input_mesh=input_mesh, scalars=["DA"], limits=[0.0, 0.50], consistent_limits=True, n_of_bar_txt_portions=6, output_type="pdf")


#For getting the p-value thresholds in the same format.
stats_dir = pathlib.Path(r"D:\Desktop\Canids\Point_cloud\Femur\BVTV")
os.chdir(stats_dir)
stats_mesh = pv.read("Femur_consolidated_ttest.vtk")

get_ttest_screens(input_mesh=stats_mesh,
                  scalars=["BVTV", "DA"],
                  estimate_limits=True,
                  n_of_bar_txt_portions=11,
                  output_type="pdf")

get_ttest_screens(input_mesh=stats_mesh, scalars=["BVTV"], estimate_limits=True, output_type="png")
get_ttest_screens(input_mesh=stats_mesh, scalars=["DA"], estimate_limits=True, output_type="png")
