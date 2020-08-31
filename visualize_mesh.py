'''

from matplotlib.colors import rgb2hex
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

def get_scalar_screens(input_mesh, scalars=["BVTV", "DA"]):
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

        for average_scalar in average_list:
            generate_plot(input_mesh=input_mesh,
                          scalar=scalar,
                          scalar_value=average_scalar,
                          scalar_type="average",
                          colormap=scalar_color_dict[str(scalar)],
                          limits=[0, 0.50])

        for coef in coef_list:
            range_max = float(input_mesh.point_arrays[str(coef)].max())
            range_min = float(input_mesh.point_arrays[str(coef)].min())
            generate_plot(input_mesh=input_mesh,
                          scalar=scalar,
                          scalar_value=coef,
                          scalar_type="Coeff. Var.",
                          colormap=scalar_color_dict["coef"],
                          limits=[range_min, range_max])

        for std in std_list:
            range_max = float(input_mesh.point_arrays[str(std)].max())
            range_min = float(input_mesh.point_arrays[str(std)].min())
            generate_plot(input_mesh=input_mesh,
                          scalar=scalar,
                          scalar_value=std,
                          scalar_type="Std. Dev.",
                          colormap=scalar_color_dict["std"],
                          limits=[range_min, range_max])


def generate_plot(input_mesh, scalar, scalar_value, scalar_type, colormap, limits=[0, 0.50]):

        scalar = scalar
        group = scalar_value.split("_", 1)[0]
        scalar_name = scalar_value

        #Set up the arguments to be passed to pyvista
        vel_dargs = dict(scalars=f"{scalar_name}", clim=limits, cmap=colormap,
                         log_scale=False, reset_camera=False)
        f_size = 15

        sargs = dict(title_font_size=1, label_font_size=25, shadow=False, n_labels=6, vertical=True,
                     position_x=0.05, position_y=0.2,
                     height=0.65, width=0.1, italic=False, fmt="%.2f", font_family="arial")

        plotter = pv.Plotter(shape=(3, 3), window_size=(3840, 2160))
        plotter.disable_parallel_projection()
        plotter.reset_camera()

        plotter.subplot(0, 0)
        plotter.add_mesh(input_mesh, **vel_dargs, scalar_bar_args=sargs)
        plotter.view_zy()
        current_pos = plotter.camera_position
        print(current_pos)
        new_position = [current_pos[0], current_pos[1], (0.0, 0.0, 1.0)]
        print(new_position)
        plotter.camera_position = new_position
        plotter.add_text("Dorsal", font_size=f_size)

        plotter.subplot(1, 0)
        plotter.add_mesh(input_mesh, **vel_dargs)
        plotter.view_yz()
        plotter.add_text("Anterior", font_size=f_size)


        plotter.subplot(0, 1)
        plotter.add_mesh(input_mesh, **vel_dargs)
        plotter.view_zx()
        current_pos = plotter.camera_position
        print(current_pos)
        r = R.from_euler('xyz', [0.0, 0.0, 0.0], degrees=True)
        new_position = tuple(np.dot(np.array(r.as_matrix()), np.array(current_pos[0])))
        plotter.camera_position = [new_position, current_pos[1], (0.0, 0.0, 1.0)]
        print(plotter.camera_position)
        plotter.add_text("Medial", font_size=f_size)

        plotter.subplot(1, 1)
        plotter.add_mesh(input_mesh, **vel_dargs)
        plotter.view_xz()
        plotter.add_text("Lateral", font_size=f_size)


        plotter.subplot(0, 2)
        plotter.add_mesh(input_mesh, **vel_dargs)
        plotter.view_yx()
        plotter.add_text("Distal", font_size=f_size)
        plotter.add_text(f"{group} {scalar} {scalar_type}", position='upper_right', font_size=25,
                         font=None, shadow=False, name=None, viewport=False)


        plotter.subplot(1, 2)
        plotter.add_mesh(input_mesh, **vel_dargs)
        plotter.view_yx()
        current_pos = plotter.camera_position
        print(current_pos)
        r = R.from_euler('xyz', [0.0, 180.0, 0.0], degrees=True)
        new_position = tuple(np.dot(np.array(r.as_matrix()), np.array(current_pos[0])))
        new_position = [new_position, current_pos[1], current_pos[2]]
        print(new_position)
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
        #plotter.save_graphic(screen_shot, title="PyVista Export")

        plotter.show(interactive=False, use_panel=False, screenshot=str(screen_shot))


import os
import sys
import pathlib
import numpy as np
import pyvista as pv
from matplotlib import cm
from scipy.spatial.transform import Rotation as R
from matplotlib.colors import LinearSegmentedColormap

directory = pathlib.Path(r"D:\Desktop\Canids\Point_cloud\Humerus")
os.chdir(directory)

#Set the "Dark" theme
pv.set_plot_theme("night")

#Set up custom colormap from paraview
rainbow_desat = LinearSegmentedColormap.from_list('rainbow_desat', ['#4747db', '#00005c', '#00ffff',
                                                                    '#008000', '#ffff00', '#ff6100',
                                                                    '#6b0000', '#e04d4d'], N=256)

linear_blue = LinearSegmentedColormap.from_list('linear_blue', ['#f5fffa', '#d0f5e9', '#abedde', '#8ee6d7', '#7aded2',
                                                                 '#70d4cd', '#66ccc9', '#60c4c4', '#59b5ba', '#53a7b0',
                                                                 '#4d9ba8', '#3f8b9e', '#3d7e94', '#3a708a', '#3a6785',
                                                                 '#3b5e80', '#3a5278', '#384870', '#313c66', '#292f59',
                                                                 '#22204d'], N=10)

BVTV_colors = rainbow_desat
DA_colors = cm.get_cmap(linear_blue)
Coef_colors = cm.get_cmap(name="hot", lut=10)
Standard_dev_colors = cm.get_cmap(name="PiYG_r", lut=10)

#Define the input name and read in to memory for visualization
inputMesh = "humerus_consolidated_scalars.vtk"
mesh = pv.read(inputMesh)



get_scalar_screens(input_mesh=input_mesh, scalars=["BVTV", "DA"])